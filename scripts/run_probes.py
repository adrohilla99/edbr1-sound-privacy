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
import sys
import time
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from edbr1.data.librispeech import speaker_utterances  # noqa: E402
from edbr1.data.urbansound8k import load_metadata  # noqa: E402
from edbr1.probes.frozen import FrozenEncoder, ScenePool, extract_codes  # noqa: E402
from edbr1.probes.splits import build_probe_split, speaker_utterances_with_transcripts  # noqa: E402
from edbr1.probes.train import (  # noqa: E402
    train_asr_probe,
    train_inverter_probe,
    train_speaker_probe,
)

LIBRI = PROJECT_ROOT / "data" / "raw" / "librispeech" / "LibriSpeech"
US8K = PROJECT_ROOT / "data" / "raw" / "urbansound8k" / "UrbanSound8K"


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
