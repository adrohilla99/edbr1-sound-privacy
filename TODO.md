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

- [x] Draft docs/03_methodology.md — front end, encoder, VQ + honest-bitrate, classifier, and the Phase-3 adversarial objective (overlay, GRL, train-only leak guards, sanity-vs-privacy caveat); evaluation probes remain a Phase-4 stub.
- [x] Encoder + classifier skeleton — depthwise-separable encoder E (~48K params) → classifier C; no-bottleneck control reproduces the baseline (**0.748 ± 0.058**, official 10-fold), artifact `results/us8k_encoder_20260702_181348/`
- [x] VQ bottleneck implementation — VQ-VAE bottleneck (codebook, straight-through estimator, commitment loss, perplexity/usage) + honest `bits_per_second`; **6-point utility-vs-bitrate sweep** (80–16000 bits/s, full 10-fold each) done → curve + tables in [RESULTS.md](RESULTS.md), artifact `results/us8k_vq_sweep_20260702_233340/`. Codebook collapses at every point (≤5.4% of 1024 codes used) — reported plainly. Collapse **fixed** (kmeans init + EMA + revival) → codes used 90.9–100%, curve flattens onto control (`results/us8k_vq_sweep_20260703_171719/`).
- [x] Adversarial gradient-reversal head — GRL + (N+1)-way speech adversary on the code, LibriSpeech-into-UrbanSound8K overlay (train-only, leak-guarded); **lambda sweep {0..2} at the 1000 bits/s knee**, full 10-fold → [RESULTS.md](RESULTS.md), artifact `results/us8k_adv_lambda_20260704_112525/`. GRL stable, utility-neutral (Δ≈−0.004); training adversary stayed at the no-speech floor (sanity signal, **not** a privacy result).
- [x] Probe networks (speaker-ID, ASR, inverter) — Phase 4a: independent, stronger-than-adversary probes attack the frozen code with leak-guarded dev-clean splits. Leakage table in [RESULTS.md](RESULTS.md), artifact `results/us8k_probes_20260705_011207/`. Finding: acoustic envelope + modest speaker identity leak (probes beat the Phase-3 floor); ASR fails (no words); adversarial λ=2 halves speaker top-1 (0.112→0.062) at no utility cost. Empirical lower bounds.
- [x] Pareto evaluation script — Phase 4b: test-time SNR robustness (`run_overlay_robustness.py`), ESC-50 frozen transfer (`run_esc50_probe.py`, ~18× chance), and the privacy–utility–compute frontier folded into `make_figures.py` (utility-vs-leakage, leakage-vs-bitrate per channel, robustness figures). Recommended point **1000 b/s, λ=2**: F1 0.807, speaker 1.2× chance, no words, envelope leaks. See [RESULTS.md](RESULTS.md). **Codebase now feature-complete; remaining work is dissertation writing.**

## Writing (engineering complete)

- [ ] Populate the dissertation draft's results tables/figures from [RESULTS.md](RESULTS.md) and `docs/figures/` (cite the committed `docs/figures/*.png`, not the gitignored `results/` copies).
- [ ] Write the Results chapter: utility-vs-bitrate + collapse fix (RQ1), adversarial λ (RQ2), leakage probes + content-specific finding (RQ3), robustness + ESC-50 transfer.
- [ ] Write the Discussion: the "envelope leaks, words don't, λ halves identity for free" story; recommended operating point.
- [ ] Write the Limitations chapter (MACs/s deferred, PESQ/STOI omitted in favour of LSD, from-scratch rather than pretrained ASR, small probe n, single-fold frozen encoders, empirical-lower-bound framing).
- [ ] Methods chapter: align prose with the as-built docs/03 + docs/04.
