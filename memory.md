# Dev Memory - 2026-02-19

## 0) Context
- Project: Anomaly Detection Framework (Phase 1 skeleton)
- Goal (1~2 lines): Build two self-supervised cores first (PatchTST-SSL for FDC, SwinMAE-SSL for vibration CWT) and validate on Google Colab GPU before migrating to local CUDA PC.
- Repo / Branch: `/home/userkh/workspace/ML_test/anomalydetection` (git not initialized in this folder)
- Environment: local Linux workspace (no local GPU), `python3` available (`python` alias missing), CUDA not available locally, target runtime is Colab GPU CUDA. Key libs required: `torch`, `torchvision`, `timm`, `PyWavelets`, `PyYAML`, `numpy`, `pandas`.

## 1) What changed today
### Code changes
- Added:
  - Core skeleton modules, datasets/models/trainers/inference structure
  - PatchTST-SSL data/model/trainer pipeline
  - SwinMAE-SSL data(CWT: pywt)/model/trainer pipeline
  - Unified scoring API and runbook/README
  - Config files for both streams
- Modified:
  - `PRD.md`, `TRD.md`, `Todo.md` (Decision Lock sections added)
- Removed:
  - None
- Key files:
  - `core/config.py`: YAML config loader + required key validation
  - `core/contracts.md`: frozen contracts (shape/objective/scoring/checkpoint)
  - `configs/patchtst_ssl.yaml`: PatchTST training/data/model defaults
  - `configs/swinmae_ssl.yaml`: SwinMAE + CWT(pywt) defaults
  - `datasets/fdc_dataset.py`: synthetic FDC -> window -> robust normalization dataset build
  - `datasets/transforms/fdc_normalization.py`: robust/zscore channel scaler with save/load
  - `models/patchtst/patchtst_ssl.py`: masked reconstruction model + masked MSE + scoring
  - `trainers/train_patchtst_ssl.py`: end-to-end PatchTST training script
  - `datasets/transforms/cwt.py`: Morlet CWT (`pywt`) scalogram + resize to image
  - `datasets/vib_dataset.py`: vibration window -> CWT image dataset pipeline
  - `models/swinmae/swinmae_ssl.py`: SwinMAE-style masked reconstruction + fallback encoder
  - `trainers/train_swinmae_ssl.py`: end-to-end SwinMAE training script
  - `inference/scoring.py`: unified `infer_score(batch, model, stream)` API
  - `README.md`: setup/run/Colab validation/migration instructions

### Config / CLI
- Commands I ran:
  - `mkdir -p configs core datasets/transforms models/patchtst models/swinmae trainers inference/adapters docs notebooks checkpoints artifacts`
  - `python3 -m compileall core datasets models trainers inference`
  - `python3 - << 'PY' ... load_yaml_config(...) ... PY` (config parse check)
  - `python3 - << 'PY' ... import torch, pywt ... PY` (dependency availability check)
  - `find . -maxdepth 3 -type f | sort`
- Important flags/env:
  - `device.prefer_cuda: true`
  - `device.amp: true`
  - `data.normalization: robust` (switchable to `zscore`)
  - `cwt.backend: pywt` (fixed for Phase 1)
  - Colab runtime must be set to GPU for CUDA validation

## 2) Decisions made (why)
- Decision: Code root is current repository folder.
  - Rationale: Matches current workspace and avoids extra path nesting.
  - Trade-off: If later packaging changes, path migration is needed.
- Decision: Objective fixed to masked reconstruction (both streams).
  - Rationale: Keeps MVP narrow and aligns with requested initial validation scope.
  - Trade-off: Forecasting mode is deferred.
- Decision: CWT backend fixed to `pywt`.
  - Rationale: Stable and simple Phase 1 implementation path.
  - Trade-off: Backend flexibility (`scipy`) deferred.
- Decision: Tensor contracts fixed as FDC `(B,T,C)` and vibration image `(B,3,H,W)`.
  - Rationale: Reduces integration ambiguity across dataset/model/trainer.
  - Trade-off: Any alternate internal layout needs adapters.
- Decision: Default normalization robust; easy config switch to zscore.
  - Rationale: More stable for outlier-prone industrial-like signals.
  - Trade-off: Slightly different distribution assumptions vs zscore.
- Decision: Unified scoring API as `infer_score(batch, model, stream)`.
  - Rationale: Single entry point for both model families.
  - Trade-off: Stream-specific diagnostics stay in `aux`, not top-level schema.
- Decision: Validation-first on Colab GPU, later migrate to local CUDA PC.
  - Rationale: Local notebook environment currently has no GPU.
  - Trade-off: Two runtime environments to manage.

## 3) Current status (working / broken)
- ✅ Working:
  - Project skeleton generated with clear module boundaries
  - Config files load successfully
  - Python syntax/import compile check passed for `core/datasets/models/trainers/inference`
  - Colab notebooks added: `notebooks/colab_patchtst_ssl.ipynb`, `notebooks/colab_swinmae_ssl.ipynb`
  - Inference example added: `inference/run_scoring_example.py`
- ⚠️ Known issues:
  - Local environment does not have `torch`/`pywt` installed, so training runtime not executed locally
- 🔴 Broken / TODO fix:
  - None critical in code structure; runtime verification pending in Colab GPU

## 4) How to reproduce
- Setup:
  - `pip install -r requirements.txt`
  - In Colab: Runtime -> Change runtime type -> GPU
- Run:
  - `python -m trainers.train_patchtst_ssl --config configs/patchtst_ssl.yaml`
  - `python -m trainers.train_swinmae_ssl --config configs/swinmae_ssl.yaml`
- Test:
  - `python -m compileall core datasets models trainers inference`
  - Confirm checkpoint files are created
- Expected output:
  - Logs showing train/val loss per epoch
  - Checkpoints:
    - `checkpoints/patchtst_ssl.pt`
    - `checkpoints/swinmae_ssl.pt`
  - FDC scaler artifact:
    - `artifacts/scaler_fdc.json`

## 5) Next actions (ordered)
1. [ ] Run both training scripts in Colab GPU and record smoke-run logs/checkpoints.
2. [ ] Run `python -m inference.run_scoring_example` for both streams using generated checkpoints in Colab.

## 6) Notes / links
- PR/Issue: N/A (no git repository initialized in this folder)
- Reference:
  - PatchTST repo: https://github.com/yuqinie98/PatchTST
  - PatchTST paper: https://arxiv.org/abs/2211.14730
  - Swin-MAE paper: https://arxiv.org/abs/2212.13805
  - Swin-MAE repo: https://github.com/Zian-Xu/Swin-MAE

## 7) Continuation update - 2026-02-20
- Updated docs to reference existing notebook workflow and inference example script.
- Expanded `docs/runbook.md` with Colab GPU setup, train commands, checkpoint verification, and scoring example commands.
- Added a short notebook/scoring pointer section in `README.md`.
- Replaced `notebooks/README.txt` placeholder text with links to the two Colab notebooks.
- This update supersedes the prior notebook placeholder note from 2026-02-19.

## 8) Continuation update - 2026-02-20 (tests/docs)
- Added smoke-test documentation updates in `README.md` and `docs/runbook.md` with `pytest -q`.
- Added a concise local sanity check sequence in `docs/runbook.md`: `python -m compileall core datasets models trainers inference` then `pytest -q`.
- Smoke tests are now tracked as added; remaining next actions are Colab GPU training verification and scoring runs with generated checkpoints.

## 9) Continuation update - 2026-02-20 (Phase 1.5 implementation + Colab flow)
- Implemented real-data connection core:
  - Added `datasets/contracts.md`.
  - Added readers: `datasets/readers/fdc_reader.py`, `datasets/readers/vib_reader.py`.
  - Added DQVL-lite: `dqvl/report.py`, `dqvl/fdc_rules.py`, `dqvl/vib_rules.py`.
  - Updated dataset builders with `data.source` branching:
    - `datasets/fdc_dataset.py` (`synthetic|csv|parquet`)
    - `datasets/vib_dataset.py` (`synthetic|csv|npy`)
  - Updated trainer config validation for source-dependent required keys:
    - `trainers/train_patchtst_ssl.py`
    - `trainers/train_swinmae_ssl.py`
  - Extended configs with minimal real-data and dqvl keys:
    - `configs/patchtst_ssl.yaml`
    - `configs/swinmae_ssl.yaml`
  - Added real-data smoke fixtures/tests:
    - `tests/smoke/data/fdc_dummy.csv`
    - `tests/smoke/data/vib_dummy.csv`
    - `tests/test_fdc_csv_smoke.py`
    - `tests/test_vib_csv_smoke.py`
- Updated Colab notebooks for direct Kaggle workflow:
  - `notebooks/colab_patchtst_ssl.ipynb`
  - `notebooks/colab_swinmae_ssl.ipynb`
  - Added Kaggle download/prep cells.
  - Added data check cells (including SwinMAE `fs` estimate from timestamp).
  - Training cells now use `*_real.yaml` first, with fallback to synthetic configs.
- Commit and push status:
  - Committed: `8ee3af8 feat(phase1.5): real-data pipeline + colab data flow [2026-02-20]`
  - Push completed successfully after transient network reset.

## 10) Restart checklist (next session)
1. Run `notebooks/colab_patchtst_ssl.ipynb` top-to-bottom:
   - bootstrap -> Kaggle download -> data check -> install -> GPU check -> train -> checkpoint check.
2. Run `notebooks/colab_swinmae_ssl.ipynb` top-to-bottom:
   - bootstrap -> Kaggle download -> data check.
   - read `estimated_fs...` output from the check cell.
   - update `configs/swinmae_ssl_real.yaml` `data.fs` to real sampling rate.
   - continue install -> GPU check -> train -> checkpoint check.
3. Verify DQVL reports were created:
   - `artifacts/dqvl/fdc/*.json`
   - `artifacts/dqvl/vib/*.json`
4. Run scoring smoke for both streams with generated checkpoints:
   - `python -m inference.run_scoring_example --stream patchtst --checkpoint checkpoints/patchtst_ssl.pt --config configs/patchtst_ssl_real.yaml`
   - `python -m inference.run_scoring_example --stream swinmae --checkpoint checkpoints/swinmae_ssl.pt --config configs/swinmae_ssl_real.yaml`
5. If Colab upload widget is disabled, run separate upload cell first:
   - `from google.colab import files; files.upload()`
