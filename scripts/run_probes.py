"""Phase-4a: run the three speech-leakage probes against each frozen encoder.

For every encoder in the probe-encoder manifest: freeze it, build leak-guarded
probe splits over held-out dev-clean speakers (disjoint from the encoder's
speakers), extract codes (speech overlaid on held-out scenes), and train three
independent, stronger-than-training-adversary probes -- speaker-ID (top-1),
CTC ASR (WER/CER), and a mel inverter (LSD/MSE). Writes an incremental leakage
table and a few reconstruction spectrograms.

Usage:
    python -u scripts/run_probes.py --device auto
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch import nn  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from edbr1.data.librispeech import speaker_utterances  # noqa: E402
from edbr1.data.urbansound8k import load_metadata  # noqa: E402
from edbr1.probes.frozen import FrozenEncoder, ProbeCodes, ScenePool, extract_codes  # noqa: E402
from edbr1.probes.metrics import (  # noqa: E402
    character_error_rate,
    log_spectral_distance,
    top1_accuracy,
    word_error_rate,
)
from edbr1.probes.models import (  # noqa: E402
    CharVocab,
    CTCProbe,
    InverterProbe,
    SpeakerProbe,
    num_parameters,
)
from edbr1.probes.splits import build_probe_split, speaker_utterances_with_transcripts  # noqa: E402

LIBRI = PROJECT_ROOT / "data" / "raw" / "librispeech" / "LibriSpeech"
US8K = PROJECT_ROOT / "data" / "raw" / "urbansound8k" / "UrbanSound8K"


def _batches(n: int, bs: int, device: torch.device) -> list[torch.Tensor]:
    perm = torch.randperm(n, device=device)
    return [perm[i:i + bs] for i in range(0, n, bs)]


def train_speaker_probe(
    codebook_size: int, tr: ProbeCodes, te: ProbeCodes, num_speakers: int,
    device: torch.device, *, epochs: int, seed: int,
) -> dict[str, Any]:
    """Closed-set speaker-ID probe -> top-1 vs chance (1/N)."""
    torch.manual_seed(seed)
    probe = SpeakerProbe(codebook_size, num_speakers).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=1e-3, weight_decay=1e-4)
    x_tr, y_tr = tr.indices.to(device), tr.speaker_labels.to(device)
    x_te, y_te = te.indices.to(device), te.speaker_labels.to(device)
    for _ in range(epochs):
        probe.train()
        for idx in _batches(len(x_tr), 128, device):
            opt.zero_grad()
            F.cross_entropy(probe(x_tr[idx]), y_tr[idx]).backward()
            opt.step()
    probe.eval()
    with torch.no_grad():
        preds = probe(x_te).argmax(dim=1)
    return {
        "top1": top1_accuracy(preds.cpu(), y_te.cpu()),
        "chance": 1.0 / num_speakers,
        "n_test": int(len(y_te)),
        "probe_params": num_parameters(probe),
    }


def train_asr_probe(
    codebook_size: int, latent_freq: int, tr: ProbeCodes, te: ProbeCodes,
    device: torch.device, *, epochs: int, seed: int,
) -> dict[str, Any]:
    """From-scratch CTC ASR probe on codes -> WER/CER vs the ~1.0 ceiling."""
    torch.manual_seed(seed)
    random.seed(seed)
    vocab = CharVocab()
    frames = tr.indices.shape[2]

    def prep(codes: ProbeCodes) -> list[tuple[torch.Tensor, list[int], str]]:
        out = []
        for i, txt in enumerate(codes.transcripts):
            enc = vocab.encode(txt)
            if 0 < len(enc) < frames:  # CTC needs input_len > target_len
                out.append((codes.indices[i], enc, txt))
        return out

    train_items, test_items = prep(tr), prep(te)
    if len(train_items) < 16 or len(test_items) < 8:
        return {"skipped": f"too few utts fit T'={frames} (tr={len(train_items)}, "
                f"te={len(test_items)})", "input_frames": frames}

    probe = CTCProbe(codebook_size, vocab.size, latent_freq).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=1e-3)
    ctc = nn.CTCLoss(blank=vocab.blank, zero_infinity=True)
    for _ in range(epochs):
        probe.train()
        random.shuffle(train_items)
        for i in range(0, len(train_items), 32):
            batch = train_items[i:i + 32]
            x = torch.stack([b[0] for b in batch]).to(device)
            logp = probe(x).permute(1, 0, 2)  # (T', B, vocab) for CTC
            targets = torch.tensor([c for b in batch for c in b[1]], device=device)
            tgt_len = torch.tensor([len(b[1]) for b in batch])
            in_len = torch.full((len(batch),), frames)
            opt.zero_grad()
            ctc(logp, targets, in_len, tgt_len).backward()
            opt.step()
    probe.eval()
    refs, hyps = [], []
    with torch.no_grad():
        for idx, _enc, txt in test_items:
            logp = probe(idx.unsqueeze(0).to(device))[0]
            hyps.append(vocab.greedy_decode(logp))
            refs.append(txt)
    return {
        "wer": word_error_rate(refs, hyps),
        "cer": character_error_rate(refs, hyps),
        "ceiling_wer": 1.0,
        "n_test": len(test_items),
        "input_frames": frames,
        "probe_params": num_parameters(probe),
    }


def train_inverter_probe(
    codebook_size: int, tr: ProbeCodes, te: ProbeCodes, silence_mel: torch.Tensor,
    device: torch.device, *, epochs: int, seed: int, fig_path: Path,
) -> dict[str, Any]:
    """Learned code->mel inverter -> LSD/MSE vs clean speech, vs a silence floor."""
    assert tr.clean_mels is not None and te.clean_mels is not None
    torch.manual_seed(seed)
    n_mels, frames = tr.clean_mels.shape[2], tr.clean_mels.shape[3]
    probe = InverterProbe(codebook_size, out_mels=n_mels, out_frames=frames).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=1e-3)
    x_tr, y_tr = tr.indices.to(device), tr.clean_mels.to(device)
    x_te, y_te = te.indices.to(device), te.clean_mels.to(device)
    for _ in range(epochs):
        probe.train()
        for idx in _batches(len(x_tr), 64, device):
            opt.zero_grad()
            F.mse_loss(probe(x_tr[idx]), y_tr[idx]).backward()
            opt.step()
    probe.eval()
    with torch.no_grad():
        recon = probe(x_te)
    lsd = log_spectral_distance(recon.cpu(), y_te.cpu())
    mse = float(F.mse_loss(recon, y_te))
    silence = silence_mel.to(y_te).expand_as(y_te)
    _save_spectrogram_examples(y_te[:3].cpu(), recon[:3].cpu(), fig_path)
    return {
        "lsd": lsd,
        "mse": mse,
        "lsd_silence_floor": log_spectral_distance(silence.cpu(), y_te.cpu()),
        "n_test": int(len(x_te)),
        "probe_params": num_parameters(probe),
        "note": "PESQ/STOI omitted (packages absent; would need waveform reconstruction)",
    }


def _save_spectrogram_examples(clean: torch.Tensor, recon: torch.Tensor, path: Path) -> None:
    n = clean.shape[0]
    fig, axes = plt.subplots(2, n, figsize=(3 * n, 4))
    for j in range(n):
        axes[0, j].imshow(clean[j, 0], origin="lower", aspect="auto")
        axes[0, j].set_title("clean speech" if j == 0 else "")
        axes[1, j].imshow(recon[j, 0], origin="lower", aspect="auto")
        axes[1, j].set_title("inverter recon" if j == 0 else "")
        for ax in (axes[0, j], axes[1, j]):
            ax.set_xticks([])
            ax.set_yticks([])
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path,
                        default=PROJECT_ROOT / "results" / "probe_encoders" / "manifest.json")
    parser.add_argument("--results-dir", type=Path, default=PROJECT_ROOT / "results")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-speakers", type=int, default=20)
    parser.add_argument("--max-utts", type=int, default=15, help="utterances/speaker cap")
    parser.add_argument("--num-scenes", type=int, default=200)
    parser.add_argument("--speaker-epochs", type=int, default=40)
    parser.add_argument("--asr-epochs", type=int, default=80)
    parser.add_argument("--inverter-epochs", type=int, default=60)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args(argv)

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    import time
    out_dir = args.results_dir / time.strftime("us8k_probes_%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Probe results: {out_dir}\nDevice: {device}")

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    us8k = load_metadata(US8K)
    fold10 = us8k[us8k["fold"] == 10].reset_index(drop=True)
    dev_index, transcripts = speaker_utterances_with_transcripts(LIBRI / "dev-clean")
    encoder_speakers = frozenset(int(s) for s in speaker_utterances(LIBRI / "train-clean-100"))

    records: list[dict[str, Any]] = []
    for name, info in manifest.items():
        print(f"\n{'=' * 60}\n{name}: {info['bits_per_second']} bits/s, lam {info['grl_lambda']}")
        frozen = FrozenEncoder(info["checkpoint"], device)
        scenes = ScenePool(fold10, num_scenes=args.num_scenes, target_len=frozen.target_len,
                           sample_rate=frozen.config.features.sample_rate, seed=args.seed)
        silence_mel = frozen.extractor(
            torch.zeros(frozen.target_len),  # CPU extractor
            frozen.config.features.sample_rate,
        ).unsqueeze(0)

        # Speaker-ID: closed set, utterance-disjoint.
        spk = build_probe_split(
            dev_index, num_speakers=args.num_speakers, mode="speaker_id",
            exclude_speakers=encoder_speakers, transcripts=transcripts,
            max_utts_per_speaker=args.max_utts, seed=args.seed,
        )
        spk_tr = extract_codes(frozen, spk.train, scenes, seed=args.seed)
        spk_te = extract_codes(frozen, spk.test, scenes, seed=args.seed + 1)
        spk_res = train_speaker_probe(frozen.codebook_size, spk_tr, spk_te, args.num_speakers,
                                      device, epochs=args.speaker_epochs, seed=args.seed)
        print(f"  speaker-ID top1 {spk_res['top1']:.3f} (chance {spk_res['chance']:.3f})")

        # ASR + inverter: speaker-disjoint (generalisation), with clean-speech mels.
        gen = build_probe_split(
            dev_index, num_speakers=args.num_speakers, mode="generalization",
            exclude_speakers=encoder_speakers, transcripts=transcripts,
            max_utts_per_speaker=args.max_utts, seed=args.seed,
        )
        gen_tr = extract_codes(frozen, gen.train, scenes, seed=args.seed + 2, want_clean_mel=True)
        gen_te = extract_codes(frozen, gen.test, scenes, seed=args.seed + 3, want_clean_mel=True)
        asr_res = train_asr_probe(frozen.codebook_size, frozen.latent_freq, gen_tr, gen_te,
                                  device, epochs=args.asr_epochs, seed=args.seed)
        print(f"  ASR WER {asr_res.get('wer', 'skipped')}")
        inv_res = train_inverter_probe(frozen.codebook_size, gen_tr, gen_te, silence_mel, device,
                                       epochs=args.inverter_epochs, seed=args.seed,
                                       fig_path=out_dir / f"{name}_inversion.png")
        floor = inv_res["lsd_silence_floor"]
        print(f"  inverter LSD {inv_res['lsd']:.2f} dB (silence floor {floor:.2f})")

        records.append({
            "name": name, "bits_per_second": info["bits_per_second"],
            "grl_lambda": info["grl_lambda"], "utility_macro_f1": info["macro_f1"],
            "codebook_perplexity": info.get("codebook_perplexity"),
            "speaker_id": spk_res, "asr": asr_res, "inverter": inv_res,
        })
        (out_dir / "leakage.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
        _write_csv(out_dir / "leakage.csv", records)

    print(f"\nProbes complete. Leakage table: {out_dir}")
    return 0


def _write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    fields = ["name", "bits_per_second", "grl_lambda", "utility_macro_f1",
              "speaker_top1", "speaker_chance", "asr_wer", "asr_cer",
              "inverter_lsd", "inverter_lsd_silence_floor"]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for r in records:
            writer.writerow({
                "name": r["name"], "bits_per_second": r["bits_per_second"],
                "grl_lambda": r["grl_lambda"], "utility_macro_f1": r["utility_macro_f1"],
                "speaker_top1": r["speaker_id"]["top1"],
                "speaker_chance": r["speaker_id"]["chance"],
                "asr_wer": r["asr"].get("wer"), "asr_cer": r["asr"].get("cer"),
                "inverter_lsd": r["inverter"]["lsd"],
                "inverter_lsd_silence_floor": r["inverter"]["lsd_silence_floor"],
            })


if __name__ == "__main__":
    raise SystemExit(main())
