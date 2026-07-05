# EDBR.1 — pre-writing completeness & integrity audit

Audit performed at commit `1186472` (Phases 2–4b reported complete). Verdicts:
**PASS** / **NEEDS-ATTENTION** / **FAIL**. Scope: verify completeness,
self-consistency, and reproducibility before switching to dissertation writing.
No results code changed and no experiments re-run; only documentation/tracking
hygiene fixed, plus a committed data snapshot of the baseline numbers.

## 1. Working tree & sync — PASS

- `git status -sb`: `## main...origin/main` — no ahead/behind, in sync.
- `git log origin/main..HEAD`: empty — no local-only commits.
- All phase commits present and pushed: `3e182dc` (baseline), `c1132de`/`005f048`/
  `a4f5cba`/`69a2b89` (Phase 2/2b), `edc8095` (Phase 3), `fea2757` (Phase 4a),
  `1186472` (Phase 4b).
- Only untracked path is `.claude/` (local editor settings; correctly not tracked).

## 2. Repository structure inventory — PASS

Root: `pyproject.toml`, `README.md`, `TODO.md`, `RESULTS.md`, `.gitignore`,
`.gitattributes`, `.python-version` — all present and tracked.

- **configs/**: `baseline.yaml`, `baseline_final.yaml`, `improved_22k.yaml`,
  `features.yaml`, `encoder_nobottleneck.yaml`, `adv/adv_lambda_base.yaml`,
  `vq/vq_{00080,00250,01000,02000,04000,16000}bps.yaml` (6). ✓
- **scripts/**: `_download_common.py`, `download_{urbansound8k,librispeech,esc50}.py`,
  `verify_environment.py`, `plot_spectrograms.py`, `make_figures.py`,
  `run_{bitrate_sweep,lambda_sweep,probes,overlay_robustness,esc50_probe}.py`,
  `train_probe_encoders.py`. ✓
- **src/edbr1/**: `config.py`, `bitrate.py`, `train.py`, `evaluate.py`,
  `utils/seed.py`; `data/{urbansound8k,librispeech,overlay,esc50,augment}.py`;
  `features/melspec.py`; `models/{encoder,bottleneck,classifier,encoder_classifier,adversary,cnn}.py`;
  `probes/{splits,frozen,models,metrics,train}.py`. ✓
- **tests/**: 14 files (config, bitrate, bottleneck, features, augment_norm, splits,
  smoke, adversary, overlay, probe_splits, probe_metrics, frozen, esc50, frontier). ✓
- **docs/**: `00_project_brief`, `01_research_questions`, `02_datasets`,
  `03_methodology`, `04_evaluation_plan`, `AUDIT` (this file), `figures/`. ✓
- **notebooks/**: `figures.ipynb`. ✓

Nothing referenced in RESULTS.md is missing from the tree.

## 3. Tests / lint / types — PASS

- `pytest`: **89 passed, 0 failed, 0 skipped** (matches the claimed 89). Zero skips
  means the data-guarded leak-guard/probe/ESC-50 tests all actually **run** (the
  datasets are present locally). *Caveat: on a clone without the datasets, the
  data-guarded tests would skip; the pure leak-guard logic tests (synthetic) run
  regardless.*
- `ruff check .`: clean.
- `mypy src/ scripts/`: clean (41 source files).

## 4. Claim-to-artifact traceability — NEEDS-ATTENTION (fixed)

`results/` is gitignored, so the committed source of truth for figures is
`docs/figures/sweep_data.json`. It contains: `control`, `collapsed` (6),
`anticollapse` (6), `lambda_sweep` (5), `leakage` (4), `robustness` (10),
`esc50` (2) — i.e. **every VQ-sweep / λ / Phase-4a leakage / robustness / ESC-50
number is preserved in committed data.**

- **Gap found:** the **baseline ablation** numbers (plain CNN 0.626; canonical
  0.746; 22.05 kHz 0.739; the 3-fold 0.698/0.711; the full 10-fold sample-rate A/B
  per-fold table) were **not** in any committed structured artifact — only in
  RESULTS.md prose and the gitignored `results/us8k_baseline_*/`.
- **Fix (this session):** committed `docs/figures/baseline_ablation.json`, a
  path-free snapshot of those numbers transcribed from RESULTS.md. Every
  dissertation-needed number now lives in a committed file.
- Note: no *figure* depends on the baseline snapshot, so it is not read by
  `make_figures.py`; it is a data-preservation record only.

## 5. Figures reproduce from committed data — PASS

- `python scripts/make_figures.py` (no `--refresh`) regenerated **all** figures
  purely from committed `sweep_data.json` (does not touch `results/`), and the
  output was **byte-identical** to the committed PNGs (git clean afterwards →
  deterministic).
- `notebooks/figures.ipynb` executes top-to-bottom via `nbconvert` (exit 0).
- Figures produced (all cited in RESULTS.md): `utility_vs_bitrate`,
  `collapsed_vs_fixed`, `codebook_usage`, `lambda_vs_utility`, `leakage_vs_bitrate`,
  `utility_vs_speaker_leakage`, `robustness_vs_snr` (+ committed
  `probe_inversion_1000bps`).
- Minor: RESULTS.md also references two *gitignored* per-run figure paths
  (`results/.../bitrate_curve.png`, `results/.../lambda_vs_utility.png`); the
  reproducible committed equivalents are `docs/figures/utility_vs_bitrate.png` and
  `docs/figures/lambda_vs_utility.png` — the dissertation should cite the
  `docs/figures/` versions.

## 6. Documentation currency — NEEDS-ATTENTION (fixed)

- **README status line** — was "Baseline pipeline in place" (stale). **Fixed** to
  "Feature-complete (Phases 2–4b)…". Quickstart commands, extras (`.[dev,ml]`),
  config names, and `verify_environment.py` are current.
- **docs/03_methodology.md** — the "Evaluation probes" section still said "Deferred
  to Phase 4 … Not implemented yet" (stale). **Fixed** to an as-built pointer to
  `edbr1.probes`, docs/04 and RESULTS. (The rest of docs/03 is as-built through
  Phase 3.)
- **docs/04_evaluation_plan.md** — as-built (privacy metrics, Pareto, robustness,
  ESC-50 all describe what was built). PASS.
- **TODO.md** — all engineering items ticked; **added** a "Writing" section with
  the genuine remaining tasks.
- **Licence** — CC BY-NC 4.0 (UrbanSound8K, ESC-50; non-commercial) stated in
  docs/02_datasets.md. PASS.

## 7. Reproducibility & integrity spot-checks — PASS

- **Seeds:** configs carry `seed` (+ overlay `seed`); `train.py` uses
  `seed_everything` and `seed_worker` (per-worker determinism).
- **Config serialised per run:** `run_dir/config.yaml` written via
  `config_to_dict`; encoder checkpoints embed the config too.
- **Leak guards exercised by non-skipped tests:** fold disjointness (`test_splits`),
  probe speaker/utterance disjointness (`test_probe_splits`), ESC-50 fold
  disjointness (`test_esc50`), frozen-encoder invariance (`test_frozen`) — all run
  (0 skips).
- **Download gating/checksums:** all three downloaders require
  `--i-have-ethics-approval`; UrbanSound8K verifies MD5 fetched live from Zenodo's
  API, LibriSpeech verifies 3 embedded OpenSLR MD5s, ESC-50 uses a structural check
  (no upstream checksum) — documented.
- **No tracked binaries:** no `.wav/.flac/.pt/.pth/.npy/.tar/.zip` tracked; largest
  tracked files are the figure PNGs (≤ 0.42 MB). No audio, model weights, or
  dataset files in git.

## 8. Honest gaps register (dissertation limitations checklist)

1. **Compute axis incomplete:** MACs/s **deferred** (no profiler dep); compute is
   reported as effective bits/s + encoder params only.
2. **Inverter metrics:** PESQ/STOI **omitted** (packages absent + need waveform
   reconstruction); reported as LSD/MSE vs a silence floor. LSD ~15–17 dB is far
   from silence but also far from codec-grade — it captures the *envelope*, not
   fine detail.
3. **ASR probe is from-scratch CTC**, not a pretrained recogniser; a pretrained
   attacker is a stronger (un-run) probe. "No words leak" is w.r.t. this probe.
4. **Small probe n:** speaker n_test = 80; ASR n_test 23–86; inverter n_test 90.
   Speaker top-1 differences across bitrate are within ~±0.03 noise; only the
   λ=0→λ=2 drop is a clean signal.
5. **Frozen encoders are single-fold retrains** of the Phase-2/3 operating points
   (fold 10 held out), not the full 10-fold set — adequate for probing, but their
   utility (0.77–0.81) is a single-split estimate.
6. **All leakage numbers are empirical LOWER bounds** — a stronger future probe can
   only raise them; nothing here proves the code is "private" in an absolute sense.
7. **Speaker identity leaks modestly (1.2–2.5× chance)** and is content-specific:
   the coarse acoustic envelope leaks heavily while linguistic content does not.
8. **ESC-50 transfer uses a light MLP head on the pooled latent** (not a strict
   linear probe); reported as such.

## Prioritised punch-list

**Fixed this session (safe doc/tracking + data snapshot):**
- [x] Committed `docs/figures/baseline_ablation.json` (Section 4 gap).
- [x] README status line → feature-complete.
- [x] docs/03 evaluation-probes stub → as-built.
- [x] TODO Writing section added.
- [x] This `docs/AUDIT.md`.

**For your decision (not changed — flag only):**
- **Baseline snapshot fidelity:** the committed baseline numbers are transcribed
  from RESULTS.md prose (the per-run `results/us8k_baseline_*/results.json` are
  gitignored). If you want a byte-exact structured snapshot, we could add a small
  `make_figures --refresh`-style extractor for the baseline runs — but that needs
  the gitignored artifacts present and is an engineering change, so deferred.
- **RESULTS.md figure citations:** a couple point at gitignored `results/.../*.png`;
  cite the `docs/figures/` equivalents in the dissertation.
- **Overall verdict:** codebase is complete, consistent, and reproducible from
  committed data. Ready for writing.
