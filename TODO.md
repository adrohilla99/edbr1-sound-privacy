# TODO

## Day 2

- [x] Submit ethics form on Qualtrics (if not already done)
- [x] Wait for waiver approval email before any download — APPROVED
- [x] Draft docs/02_datasets.md body (licensing audit, storage plan)
- [x] Implement download scripts after approval
- [x] Verify dataset checksums (Zenodo MD5 for US8K, OpenSLR MD5 for LibriSpeech; ESC-50 structural check — no upstream checksum)

## Day 3

- [x] Add `ml` optional-dependency group (torch, torchaudio, soundfile, librosa, scikit-learn, matplotlib, seaborn) — in pyproject `[ml]` extra
- [x] Mel-spectrogram preprocessing + sanity-check script on UrbanSound8K fold 1 only (scripts/plot_spectrograms.py)
- [x] Small-CNN baseline + 10-fold CV training entry point (python -m edbr1.train); **run with data to confirm the ~73–76% macro-F1 target** — DONE: regularised recipe (16 kHz, `configs/baseline_final.yaml`) reaches **0.746 ± 0.050** macro-F1 (10-fold), in the published band. 22.05 kHz A/B-tested → no gain (0.739), rejected. Canonical artifact: `results/us8k_baseline_20260627_132115/`. Full ablation + A/B table in [RESULTS.md](RESULTS.md).

## Day 4+

- [ ] Draft docs/03_methodology.md
- [ ] Encoder + classifier skeleton
- [ ] VQ bottleneck implementation
- [ ] Adversarial gradient-reversal head
- [ ] Probe networks (speaker-ID, ASR, inverter)
- [ ] Pareto evaluation script
