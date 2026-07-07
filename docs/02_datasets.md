# Datasets

This project uses three public audio datasets. Ethics waiver: **APPROVED**.
All three are fetched by the gated scripts under `scripts/`, which require
the explicit `--i-have-ethics-approval` flag and verify integrity before
use. Nothing downloaded is ever committed (see `data/README.md` and
`.gitignore`).

| Dataset       | Role                              | Clips | Classes | Official CV         | Licence            |
|---------------|-----------------------------------|-------|---------|---------------------|--------------------|
| UrbanSound8K  | Primary classification (utility)  | 8732  | 10      | 10-fold (`fold` col)| CC BY-NC 4.0       |
| LibriSpeech   | Adversarial speech-leakage probe  | —     | —       | predefined subsets  | CC BY 4.0          |
| ESC-50        | Cross-dataset generalisation      | 2000  | 50      | 5-fold              | CC BY-NC 4.0 (mixed)|

## Primary classification (UrbanSound8K)

- **Source:** https://urbansounddataset.weebly.com/urbansound8k.html
- **Mirror used:** Zenodo record [1203745](https://zenodo.org/records/1203745)
  (`UrbanSound8K.tar.gz`, ~6 GB). The downloader resolves the file and its
  **MD5** via the Zenodo REST API and verifies before extraction.
- **Split discipline:** the dataset ships an **official 10-fold** split in
  `metadata/UrbanSound8K.csv` (`fold` column). The curators warn that
  reshuffling leaks related slices (same source recording) across folds and
  inflates scores. We therefore **only** ever split on the `fold` column —
  enforced in code by `train_test_fold_split()` and its leak guard, and by
  `tests/test_splits.py`.
- **Labels:** 10 classes, `classID` 0–9 (air_conditioner, car_horn,
  children_playing, dog_bark, drilling, engine_idling, gun_shot,
  jackhammer, siren, street_music).

## Adversarial speech probe (LibriSpeech)

- **Source:** OpenSLR resource 12, https://www.openslr.org/12
- **Subsets in scope (only these):** `test-clean`, `dev-clean`,
  `train-clean-100`. The downloader **hard-refuses** any other subset
  (e.g. `train-clean-360`, `train-other-500`) at both the argparse layer
  and inside `download()` (defence in depth). Each archive is verified
  against the **MD5** published on the OpenSLR page.
- **Rationale:** LibriSpeech provides clean read speech to train/evaluate
  the speech-leakage probes later in the project. We deliberately avoid the
  larger/noisier subsets to keep storage and scope contained, and because
  the privacy threat model only needs intelligible speech, not scale.

## Cross-dataset generalisation (ESC-50)

- **Source:** https://github.com/karolpiczak/ESC-50 (master branch zip).
- **Integrity:** GitHub's generated archive is **not byte-stable**, so the
  upstream repo publishes no fixed checksum. Integrity is therefore checked
  **structurally** after extraction: the metadata CSV must be present and
  exactly **2000** `.wav` clips must be found, else the run aborts. If the
  project later needs a hash-pinned copy, switch to the Harvard Dataverse /
  Zenodo mirror and add an MD5 check (noted in the script).
- **Role:** held out for cross-dataset generalisation checks; not used to
  fit the primary classifier.

## Licensing audit

| Dataset      | Licence                          | Commercial use | Attribution | Notes |
|--------------|----------------------------------|----------------|-------------|----------------------------|
| UrbanSound8K | CC BY-NC 4.0                     | **No**         | Required    | Non-commercial only. This is academic research, which is in scope; cite Salamon, Jacoby & Bello (2014). Do **not** redistribute the audio. |
| LibriSpeech  | CC BY 4.0                        | Yes            | Required    | Most permissive of the three; cite Panayotov et al. (2015). Derived from public-domain LibriVox readings. |
| ESC-50       | CC BY-NC 4.0 (audio), mixed      | **No**         | Required    | Some clips carry their own Freesound attributions; consult the upstream `LICENSE` before publishing any clip or derivative. Cite Piczak (2015). |

**Implications.** Two of the three datasets are **non-commercial** (CC
BY-NC). This project is non-commercial academic research, so use is
permitted, but: (1) the original startup motivation is kept strictly
separate (the brief already states this), (2) no raw audio is redistributed
or committed, and (3) every dataset is cited. Any future commercial
spin-out would need to re-source UrbanSound8K and ESC-50 under different
terms or replace them.

## Storage and access

- **Layout:** `data/raw/<dataset>/…` for untouched downloads,
  `data/processed/` for derived artefacts (e.g. cached features). Both are
  gitignored; only `data/README.md` is tracked.
- **Approximate on-disk size:** UrbanSound8K ~6 GB, LibriSpeech (three
  permitted subsets) ~30 GB, ESC-50 ~0.6 GB → budget ~40 GB plus headroom
  for processed features.
- **Placement:** keep `data/` off the OS drive; mount from external/secondary
  storage locally and from a mounted datastore on Azure ML. The downloaders
  take `--data-root` so the location is configurable per machine.
- **Integrity & idempotency:** downloads land in `<name>.part` and are only
  promoted after the full stream verifies; re-running skips files already
  present and verified, and skips extraction when the marker file exists.
- **Retention:** raw archives may be deleted after extraction to reclaim
  space; re-running a downloader re-fetches and re-verifies on demand.
