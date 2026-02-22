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

## 11) Continuation update - 2026-02-21 (Colab real-data execution)
- Completed end-to-end notebook runs in Colab for both streams:
  - `notebooks/colab_patchtst_ssl.ipynb`
  - `notebooks/colab_swinmae_ssl.ipynb`
- PatchTST real-data training confirmed with epoch/loss logs:
  - Example: Epoch 1/2 and Epoch 2/2 outputs with decreasing train/val losses.
- SwinMAE real-data flow completed, including vibration dataset checks and visualization.
- Kaggle auth issue resolved by using API key/username runtime input; upload widget path was unstable in some sessions.
- Vibration `fs` handling:
  - Timestamp-derived estimate produced unrealistic value (`~1e9`) due index-like timestamp behavior.
  - Set `configs/swinmae_ssl_real.yaml` `data.fs = 10000` based on dataset context/metadata assumption.
- Added notebook-level usability improvements:
  - Training cells switched to streaming subprocess output for live progress visibility.
  - Added/used data inspection and comparison cells (head, axis checks, healthy vs fault visual checks).
- Git status:
  - Notebook updates committed and pushed.
  - Commit: `405affb chore(notebooks): refine colab execution flow [2026-02-21]`

## 12) Current blocker and next plan (2026-02-21)
- Blocker:
  - Colab GPU quota appears exhausted; runtime connection became unstable.
- Next actions after GPU becomes available:
  1. Reconnect Colab runtime and verify GPU availability (`torch.cuda.is_available()`).
  2. Re-run only minimum verification cells:
     - repo bootstrap
     - data check cells
     - training cells (PatchTST, SwinMAE)
     - checkpoint existence checks
  3. Run inference smoke for both streams with real configs/checkpoints.
  4. Review DQVL JSON outputs and summarize final keep/drop decisions.
  5. Save final run bundle (checkpoints + real configs) to persistent storage (Drive or local download).
- Fallback if GPU remains unavailable:
  1. Run CPU-only smoke by reducing epochs/batches in real configs.
  2. Defer full training until GPU quota resets.

## 13) Continuation update - 2026-02-21 (PatchTST stabilization + final artifact save)
- PatchTST real-data training instability was reproduced and diagnosed:
  - Symptom: extremely large losses (`train ~2.29e6`, `val ~3.44e4`) despite finite values.
  - Input-scale check showed normalization explosion:
    - `abs_max ~= 31057`, `abs_p99 ~= 31057`
    - scaler stats: `scale min ~= 0.000128`, `tiny_scale_count(<0.05) = 9`
- Stabilization actions applied in Colab runtime flow:
  1. Generated `swat_fdc_train_clean_v2.csv` from clean SWaT file.
  2. Preserved channel count, but handled low-variance channels with conservative median-fix and outlier clipping.
  3. Updated PatchTST real config:
     - `data.path = /content/AnomalyDetection/data/fdc/swat_fdc_train_clean_v2.csv`
     - `training.lr = 1e-4`
     - `device.amp = false`
     - `training.epochs = 10`
- Re-training result (PatchTST, real config):
  - Epoch 1: `train=8.029775`, `val=39.985680`
  - Epoch 10: `train=4.214598`, `val=21.153209`
  - Conclusion: loss explosion resolved and learning trend became stable.
- Artifact verification and save:
  - Verified presence of both checkpoints and both real configs.
  - Saved runtime bundle:
    - `/content/run_bundle_20260221_095519`
    - files: `patchtst_ssl.pt`, `swinmae_ssl.pt`, `patchtst_ssl_real.yaml`, `swinmae_ssl_real.yaml`
- Current phase status:
  - Phase-1 objective (train + smoke-level readiness + artifact persistence) is complete.
  - Threshold policy (TPR/FPR operating point) intentionally deferred to next phase.

## 14) Continuation update - 2026-02-22 (Phase 2 dashboard bootstrap + T02 export)
- Planning and loop setup updates:
  - Rebuilt `IMPLEMENTATION_PLAN.md` to make Priority Queue and Task Cards 1:1.
  - Added explicit per-task Acceptance Criteria / Deliverables / Verify.
  - Clarified in-scope/out-of-scope and open decision questions with defaults.
- Training output validation pipeline enhancements:
  - Added post-training checklist automation in `pipelines/validate_training_outputs.py` (already present and reused).
  - Added loss artifact persistence in trainers (`artifacts/loss/*_loss_history.csv`, `*_loss_curve.png`).
- Dashboard T01 implemented (skeleton):
  - Added `training_dashboard/` static app:
    - `index.html`
    - `css/main.css`, `css/nodes.css`, `css/animations.css`
    - `js/app.js`, `js/nodes.js`, `js/connections.js`, `js/drag.js`
    - `data/dashboard-layout.json`
  - Verified rendering route:
    - `python -m http.server 8765 --directory /home/userkh/workspace/ML_test/anomalydetection/training_dashboard`
    - URL: `http://127.0.0.1:8765/index.html`
- Dashboard T02 implemented (runtime state export):
  - Added `pipelines/export_training_dashboard_state.py`.
  - Added schema validation for state contract:
    - top-level keys: `meta`, `nodes`, `checklist`, `metrics`, `artifacts`.
  - Reused existing validator checks (no duplicated logic).
  - Added test file:
    - `tests/pipelines/test_export_training_dashboard_state.py`
  - Updated docs:
    - `docs/runbook.md` with Phase 2 export commands.
  - Updated ignore:
    - `.gitignore` includes `training_dashboard/data/dashboard-state.json`.
- Environment and verification notes:
  - Local run of export works and generates `dashboard-state.json`.
  - Local checklist may show partial pass (e.g., `3/7`) when training artifacts are not present locally.
  - Colab checklist previously reached full pass (`7/7`) when artifacts existed.
  - No critical code error found for T02; observed differences were environment/path dependent (local vs Colab artifact presence).
- Local venv safety cleanup:
  - User requested removing only today's installed packages from `.venv`.
  - Generated "today package list" and uninstalled those packages only.
  - Confirmed heavy CUDA-related pip packages were removed from `.venv`.
  - Kept `.venv` directory itself.

## 15) Git commits summary - 2026-02-22
- `45a3c25` chore(ralph-loop): bootstrap phase2 dashboard planning baseline
- `4832d1f` chore(plan): reprioritize ralph loop for dashboard p0 flow
- `efa3c82` chore(plan): harden dashboard execution criteria
- `ab68f10` feat(training-dashboard): scaffold p0 dashboard skeleton
- `c183c50` feat(pipelines): add dashboard runtime state export
- `c4a8814` chore(notebooks): sync colab swinmae session state
- Note:
  - `c731ff3` (`chore(wip): ralph checkpoint ...`) exists as auto WIP commit from loop script behavior.

## 16) Remaining work (next execution order)
1. T02 - Colab environment result confirmation
   - Run checklist in Colab and save evidence log:
     - `python -m pipelines.validate_training_outputs --repo-root /content/AnomalyDetection --smoke | tee /content/AnomalyDetection/artifacts/validation/validate_<date>.txt`
   - Rebuild runtime state in Colab and confirm `dashboard-state.json`:
     - `python -m pipelines.export_training_dashboard_state --repo-root /content/AnomalyDetection --out /content/AnomalyDetection/training_dashboard/data/dashboard-state.json --run-id <run_id> --run-smoke`
   - Confirm top-level contract keys exist: `meta`, `nodes`, `checklist`, `metrics`, `artifacts`.
2. T03 - Checklist and metrics panels in dashboard UI
   - Make frontend consume `training_dashboard/data/dashboard-state.json`.
   - Render checklist rows (`[v]/[ ]`, PASS/FAIL, detail/hint).
   - Render summary cards (checkpoints/scaler/logs/backup readiness).
   - Render loss trends for PatchTST/SwinMAE from exported state.
3. T04 - Run history and comparison
   - Persist per-run snapshots and selector UI.
   - Add baseline delta comparison (checklist pass count and final losses).
4. T05 - UX polish
   - Finalize status animation by node state.
   - Add quick links to artifact/log paths.
   - Confirm mobile-safe layout behavior.
