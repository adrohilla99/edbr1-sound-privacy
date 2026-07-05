"""Phase-4b Task A: test-time speech-over-scene SNR robustness at the knee.

For the frozen 1000 bits/s encoders (lambda 0 and lambda 2), sweep the test-time
speech-to-scene SNR and report, at each SNR:
  * utility  -- macro-F1 of the frozen scene classifier on held-out (fold-10)
    UrbanSound8K scenes overlaid with dev-clean speech at that SNR;
  * leakage  -- the three probes (speaker-ID top-1, ASR WER, inverter LSD) on
    dev-clean speech overlaid at that SNR.
Leak-guarded exactly as Phase 4a: dev-clean probe speakers are disjoint from the
encoder's train-clean-100 speakers, and the scenes are the encoder's held-out
fold. This maps the "loud argument in the street" condition.

Usage:
    python -u scripts/run_overlay_robustness.py --wav-cache data/processed/wavcache
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import soundfile as sf
import torch
import torchaudio.functional as AF

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from edbr1.data.librispeech import SpeechPool, speaker_utterances  # noqa: E402
from edbr1.data.overlay import mix_at_snr  # noqa: E402
from edbr1.data.urbansound8k import load_metadata  # noqa: E402
from edbr1.evaluate import classification_metrics  # noqa: E402
from edbr1.probes.frozen import FrozenEncoder, ScenePool, extract_codes  # noqa: E402
from edbr1.probes.splits import build_probe_split, speaker_utterances_with_transcripts  # noqa: E402
from edbr1.probes.train import (  # noqa: E402
    train_asr_probe,
    train_inverter_probe,
    train_speaker_probe,
)

LIBRI = PROJECT_ROOT / "data" / "raw" / "librispeech" / "LibriSpeech"
US8K = PROJECT_ROOT / "data" / "raw" / "urbansound8k" / "UrbanSound8K"
DEFAULT_SNRS = [-10.0, -5.0, 0.0, 5.0, 10.0]
KNEE = ("bps01000_l0", "bps01000_l2")  # 1000 bits/s, lambda 0 and 2


def _fix(wave: torch.Tensor, n: int) -> torch.Tensor:
    if wave.shape[-1] >= n:
        return wave[:n]
    return torch.nn.functional.pad(wave, (0, n - wave.shape[-1]))


def preload_scenes(meta: Any, target_len: int, sr: int, max_clips: int, seed: int):
    """Load up to ``max_clips`` fold-10 scene waveforms + their US8K labels."""
    rows = meta.sample(n=min(max_clips, len(meta)), random_state=seed)
    waves, labels = [], []
    for path, cls in zip(rows["path"], rows["classID"], strict=True):
        data, file_sr = sf.read(str(path), dtype="float32", always_2d=True)
        mono = torch.from_numpy(data).mean(dim=1)
        if file_sr != sr:
            mono = AF.resample(mono, file_sr, sr)
        waves.append(_fix(mono, target_len))
        labels.append(int(cls))
    return torch.stack(waves), labels


def utility_under_overlay(
    frozen: FrozenEncoder, scene_waves: torch.Tensor, scene_labels: list[int],
    speech_pool: SpeechPool, snr: float, class_names: list[str], *, seed: int,
) -> float:
    """Macro-F1 of the frozen classifier on scenes overlaid with speech at ``snr``."""
    gen = torch.Generator().manual_seed(seed)
    sr = frozen.config.features.sample_rate
    preds: list[int] = []
    for i in range(0, len(scene_waves), 64):
        mels = []
        for w in scene_waves[i:i + 64]:
            speech, _ = speech_pool.sample(gen)
            mels.append(frozen.extractor(mix_at_snr(w, speech, snr), sr).unsqueeze(0))
        preds.extend(frozen.classify(torch.stack(mels)).argmax(dim=1).cpu().tolist())
    return float(classification_metrics(scene_labels, preds, class_names)["macro_f1"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path,
                        default=PROJECT_ROOT / "results" / "probe_encoders" / "manifest.json")
    parser.add_argument("--results-dir", type=Path, default=PROJECT_ROOT / "results")
    parser.add_argument("--wav-cache", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--snrs", type=float, nargs="+", default=DEFAULT_SNRS)
    parser.add_argument("--num-speakers", type=int, default=20)
    parser.add_argument("--max-utts", type=int, default=15)
    parser.add_argument("--num-scenes", type=int, default=200)
    parser.add_argument("--utility-clips", type=int, default=400)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args(argv)

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    out_dir = args.results_dir / time.strftime("us8k_robustness_%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Robustness results: {out_dir}\nDevice: {device}\nSNRs: {args.snrs} dB")

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    from edbr1.data.urbansound8k import URBANSOUND8K_CLASSES
    us8k = load_metadata(US8K)
    fold10 = us8k[us8k["fold"] == 10].reset_index(drop=True)
    dev_index, transcripts = speaker_utterances_with_transcripts(LIBRI / "dev-clean")
    encoder_speakers = frozenset(int(s) for s in speaker_utterances(LIBRI / "train-clean-100"))

    records: list[dict[str, Any]] = []
    for name in KNEE:
        info = manifest[name]
        frozen = FrozenEncoder(info["checkpoint"], device)
        sr = frozen.config.features.sample_rate
        scene_waves, scene_labels = preload_scenes(fold10, frozen.target_len, sr,
                                                    args.utility_clips, args.seed)
        scenes = ScenePool(fold10, num_scenes=args.num_scenes, target_len=frozen.target_len,
                           sample_rate=sr, seed=args.seed)
        speech_pool = SpeechPool(LIBRI, subset="dev-clean", num_speakers=args.num_speakers,
                                 segments_per_speaker=args.max_utts, segment_seconds=4.0,
                                 sample_rate=sr, seed=args.seed, cache_dir=args.wav_cache)
        silence_mel = frozen.extractor(torch.zeros(frozen.target_len), sr).unsqueeze(0)
        spk = build_probe_split(dev_index, num_speakers=args.num_speakers, mode="speaker_id",
                                exclude_speakers=encoder_speakers, transcripts=transcripts,
                                max_utts_per_speaker=args.max_utts, seed=args.seed)
        gen = build_probe_split(dev_index, num_speakers=args.num_speakers, mode="generalization",
                                exclude_speakers=encoder_speakers, transcripts=transcripts,
                                max_utts_per_speaker=args.max_utts, seed=args.seed)

        for snr in args.snrs:
            util = utility_under_overlay(frozen, scene_waves, scene_labels, speech_pool, snr,
                                         list(URBANSOUND8K_CLASSES), seed=args.seed)
            spk_tr = extract_codes(frozen, spk.train, scenes, snr_choices=(snr,), seed=args.seed)
            spk_te = extract_codes(frozen, spk.test, scenes, snr_choices=(snr,), seed=args.seed + 1)
            spk_res = train_speaker_probe(frozen.codebook_size, spk_tr, spk_te, args.num_speakers,
                                          device, epochs=40, seed=args.seed)
            gen_tr = extract_codes(frozen, gen.train, scenes, snr_choices=(snr,),
                                   seed=args.seed + 2, want_clean_mel=True)
            gen_te = extract_codes(frozen, gen.test, scenes, snr_choices=(snr,),
                                   seed=args.seed + 3, want_clean_mel=True)
            asr_res = train_asr_probe(frozen.codebook_size, frozen.latent_freq, gen_tr, gen_te,
                                      device, epochs=80, seed=args.seed)
            inv_res = train_inverter_probe(frozen.codebook_size, gen_tr, gen_te, silence_mel,
                                           device, epochs=60, seed=args.seed)
            rec = {
                "name": name, "grl_lambda": info["grl_lambda"], "snr_db": snr,
                "utility_macro_f1": util, "speaker_top1": spk_res["top1"],
                "speaker_chance": spk_res["chance"], "asr_wer": asr_res.get("wer"),
                "inverter_lsd": inv_res["lsd"],
                "inverter_lsd_silence_floor": inv_res["lsd_silence_floor"],
            }
            records.append(rec)
            print(f"  {name} SNR {snr:+.0f}: util {util:.3f} | spk {spk_res['top1']:.3f} "
                  f"| WER {asr_res.get('wer', 'skip')} | inv LSD {inv_res['lsd']:.1f}")
            (out_dir / "robustness.json").write_text(
                json.dumps(records, indent=2), encoding="utf-8")
            _write_csv(out_dir / "robustness.csv", records)

    print(f"\nRobustness sweep complete: {out_dir}")
    return 0


def _write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    fields = ["name", "grl_lambda", "snr_db", "utility_macro_f1", "speaker_top1",
              "speaker_chance", "asr_wer", "inverter_lsd", "inverter_lsd_silence_floor"]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


if __name__ == "__main__":
    raise SystemExit(main())
