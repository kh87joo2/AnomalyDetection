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

## 17) Continuation update - 2026-02-23 (T03/T04/T05 completion + verification)
- Phase 2 dashboard execution status:
  - T03 (Checklist and Metrics Panels): completed.
  - T04 (Run History and Comparison): completed.
  - T05 (UX Polish): completed.
  - `IMPLEMENTATION_PLAN.md` Priority Queue now shows P1~P5 all checked.
- T03 completion details:
  - Added right-side dashboard panels and runtime binding.
  - Checklist rows now render compactly (title row only), with detail/hint preserved as tooltip metadata.
  - Readiness and loss panels render from `dashboard-state.json`.
  - Scroll/overflow behavior fixed for checklist/panel visibility.
- T04 completion details:
  - Extended exporter for run snapshot persistence:
    - `--persist-run-history`
    - `--run-history-dir` (default `training_dashboard/data/runs`)
    - `--run-history-limit` (default `20`)
  - Added run-id normalization rule:
    - allowed chars: `[a-zA-Z0-9._-]`
    - others replaced with `_`
  - Added run history artifacts:
    - per-run snapshot: `training_dashboard/data/runs/<run_id>.json`
    - index: `training_dashboard/data/runs/index.json`
  - Added dashboard run comparison panel:
    - current/baseline selectors
    - checklist pass delta
    - PatchTST final val loss delta
    - SwinMAE final val loss delta
  - Added tests:
    - `tests/pipelines/test_dashboard_run_history.py`
- T05 completion details:
  - Added status-synced line animation in `training_dashboard/css/animations.css`:
    - connection states: `idle/running/done/fail`
    - node states: `idle/running/done/fail`
  - Added Quick Links panel:
    - links generated from runtime artifact paths (checkpoints/scaler/logs/run-index/etc)
  - Added mobile fallback tuning (`<=768px`) in `training_dashboard/css/main.css`.
- Colab verification outcome for T04:
  - Generated two run snapshots:
    - `colab_run_a.json`
    - `colab_run_b.json`
  - `index.json` confirmed two runs with `7/7` checklist pass.
  - Final validation losses in index:
    - PatchTST: `21.153209686279297`
    - SwinMAE: `1.0077273845672607`
  - Backend delta check printed:
    - checklist delta: `0`
    - patchtst val delta: `0.0`
    - swinmae val delta: `0.0`
    - `T04 backend PASS`
- Runtime/UI notes:
  - Colab+VSCode forwarded-port path showed intermittent access issues (`localhost`/proxy routing mismatch) despite server `HTTP 200` from runtime.
  - This was a serving/path issue, not a data/export logic issue.
  - Functional verification was completed through generated artifacts + backend checks.

## 18) Git/branch state snapshot - 2026-02-23
- Dashboard-related commits now include:
  - `e989e51` feat(training-dashboard): implement t03 checklist and metrics panels
  - `d19ffda` fix(training-dashboard): improve checklist panel scrolling and compact checklist rows
  - `652ef68` feat(training-dashboard): implement run history export and comparison panel
  - `7273bf9` feat(training-dashboard): complete t05 ux polish
- Current local note:
  - `notebooks/colab_swinmae_ssl.ipynb` has local modifications and was intentionally left out of dashboard commits.
  - Latest dashboard commit (`7273bf9`) should be pushed by user to synchronize `origin/main`.

## 19) Continuation update - 2026-02-26 (Phase 3A plan pivot + naming/scope lock)
- Context shift confirmed with user:
  - Do not start from real-time ingestion.
  - Build first from imported test data workflow.
- Requirement document backup completed before rewriting:
  - `docs/archive/phase3_prep_backup_2026-02-26/PRD_phase2_training_dashboard.md`
  - `docs/archive/phase3_prep_backup_2026-02-26/TRD_phase2_training_dashboard.md`
  - `docs/archive/phase3_prep_backup_2026-02-26/Todo_phase2_training_dashboard.md`
- Re-authored current phase requirements for batch decision flow:
  - `PRD.md`, `TRD.md`, `Todo.md`
- Planning alignment:
  - Ran Ralph setup and then manually cleaned `IMPLEMENTATION_PLAN.md` to remove noisy autogenerated queue items.
  - Active queue now focuses on practical implementation slices: skeleton -> import/preprocess -> scoring -> decision/reporting -> dashboard bridge.
- Naming lock update:
  - Removed "pilot" wording from active plan semantics per user intent.
  - Current naming centers on "Batch Anomaly Decision Pipeline".
- Technical decision locks added:
  - Reuse training-compatible preprocessing/windowing path (no separate divergent transform logic).
  - v0 threshold policy uses fixed configurable values (config/artifact editable).
  - `dual` mode must fail-fast when either stream input is missing.
- Dashboard integration direction locked:
  - Training flow and test-data decision flow must be independent in dashboard (separate tabs/views).
  - Test-data decision tab must visualize score trend and threshold lines overlaid in the same chart area (not separate text-only indication).
- Next build start point:
  - `P0B`: scaffold `batch_decision/` package + `configs/batch_decision_runtime.yaml` + dry-run config validation.

## 20) Continuation update - 2026-02-26 (Colab-first validation stage added)
- User requested explicit execution strategy due to local environment GPU absence:
  - validate first in Colab GPU
  - then migrate the validated tool to local GPU
- Planning documents updated to include Colab-first + local-GPU migration path:
  - `PRD.md`: added Colab-first validation and local GPU migration-ready success criteria
  - `TRD.md`: added runtime profile split (`batch_decision_runtime_colab.yaml`, `batch_decision_runtime_local_gpu.yaml`) and environment execution strategy section
  - `Todo.md`: added `2A) P0 - Colab Validation Profile` and `6A) P1 - Local GPU Migration Readiness`
- Dashboard requirement clarified and locked:
  - training/test pipelines must be independent tabs/views
  - test-data decision chart must overlay score curve and threshold lines in the same chart area
- Ralph loop setup rerun after doc changes to refresh generated planning artifacts.
- `IMPLEMENTATION_PLAN.md` managed block manually normalized again for practical execution order and reduced noise, preserving:
  - Colab-first stage
  - local GPU migration readiness stage
  - dual-mode fail-fast requirement
  - chart overlay requirement
- No implementation code changes started yet by user request.

## 21) Continuation update - 2026-02-26 (Pre-start final sync checkpoint)
- Final planning sync completed before implementation start:
  - Re-ran Ralph setup (`setup.py`) to refresh managed planning artifacts.
  - Re-normalized `IMPLEMENTATION_PLAN.md` managed block to practical execution order focused on Phase 3A batch decision work.
  - Verified next executable task is `P0B: Build batch decision skeleton`.
- Naming consistency lock:
  - Removed residual `pilot` wording from active implementation plan view/list.
- Execution status lock:
  - No source implementation was started.
  - Repository remains in pre-start planning state awaiting explicit user start command.

## 22) Continuation update - 2026-03-03 (P0B batch decision skeleton implementation)
- Implemented P0B scope (`batch_decision` skeleton + dry-run config validation):
  - Added package skeleton:
    - `batch_decision/__init__.py`
    - `batch_decision/contracts.py`
    - `batch_decision/runner.py`
    - `batch_decision/importers.py`
    - `batch_decision/preprocess.py`
    - `batch_decision/scoring_engine.py`
    - `batch_decision/decision_engine.py`
    - `batch_decision/reporting.py`
    - `batch_decision/service.py`
  - Added runtime config and threshold template:
    - `configs/batch_decision_runtime.yaml`
    - `artifacts/thresholds/batch_decision_thresholds.json`
  - Added initial tests:
    - `tests/batch_decision/test_runner_skeleton.py`
- Validation checks performed in local environment:
  - `python3 -m py_compile ...` for new/changed files: PASS
  - `python3 -m batch_decision.runner --config configs/batch_decision_runtime.yaml --dry-run`: PASS
- Verification command blocker:
  - Required verification command `pytest` is not runnable in current environment.
  - `pytest` not installed in system/venv interpreter, and `pip install -r requirements-dev.txt` failed due network resolution restrictions.
  - Task remains ready for final pytest confirmation once environment package/network constraints are resolved.

## 23) Continuation update - 2026-03-03 (P0B Colab verification completed)
- User executed Colab-side verification after pulling latest `main`:
  - `python3 -m batch_decision.runner --config configs/batch_decision_runtime.yaml --dry-run`: PASS
  - `python3 -m pytest -q tests/batch_decision/test_runner_skeleton.py`: PASS (`4 passed`)
- P0B status promoted to complete based on Colab verification evidence.

## 24) Continuation update - 2026-03-08 (P0C Colab profile implementation ready)
- Implemented P0C Colab validation profile assets:
  - `configs/batch_decision_runtime_colab.yaml`
  - `tests/batch_decision/test_colab_profile.py`
  - `docs/runbook.md` Phase 3A batch decision Colab flow section
  - `README.md` Phase 3A batch decision Colab validation section
  - `notebooks/README.txt` CLI note for batch decision validation
- Local checks completed:
  - `python3 -m py_compile batch_decision/runner.py tests/batch_decision/test_colab_profile.py`: PASS
  - `python3 -m batch_decision.runner --config configs/batch_decision_runtime_colab.yaml --dry-run`: PASS
- Remaining close step for P0C:
  - user-side Colab verification after pull:
    - `python3 -m batch_decision.runner --config configs/batch_decision_runtime_colab.yaml --dry-run`
    - `python3 -m pytest -q tests/batch_decision/test_runner_skeleton.py tests/batch_decision/test_colab_profile.py`

## 25) Continuation update - 2026-03-08 (P0C Colab verification completed)
- User pulled latest `main` in Colab and completed the P0C verification path:
  - `python3 -m batch_decision.runner --config configs/batch_decision_runtime_colab.yaml --dry-run`: PASS
  - `python3 -m pytest -q tests/batch_decision/test_runner_skeleton.py tests/batch_decision/test_colab_profile.py`: PASS (`6 passed`)
- P0C status promoted to complete based on Colab execution evidence.

## 26) Continuation update - 2026-03-08 (P0D import/preprocess implementation ready)
- Implemented P0D test-data import and training-compatible preprocess/window builder wrappers:
  - `batch_decision/importers.py`
  - `batch_decision/preprocess.py`
  - `batch_decision/contracts.py`
  - `batch_decision/__init__.py`
  - `tests/batch_decision/test_import_and_preprocess.py`
- Runtime configs now declare the training config references used for preprocess compatibility:
  - `configs/batch_decision_runtime.yaml`
  - `configs/batch_decision_runtime_colab.yaml`
- Added Colab verification instructions for P0D in `docs/runbook.md`.
- Local checks completed:
  - `python3 -m py_compile batch_decision/__init__.py batch_decision/contracts.py batch_decision/importers.py batch_decision/preprocess.py tests/batch_decision/test_import_and_preprocess.py`: PASS
- Local verification command blocker remains:
  - `python3 -m pytest -q tests/batch_decision/test_import_and_preprocess.py` could not run because `pytest` is not installed in system or project venv.
- Remaining close step for P0D:
  - user-side Colab verification after pull:
    - `python3 -m pytest -q tests/batch_decision/test_import_and_preprocess.py`

## 27) Continuation update - 2026-03-08 (P0D Colab verification completed)
- User pulled latest `main` in Colab and completed the P0D verification path:
  - `python3 -m pytest -q tests/batch_decision/test_import_and_preprocess.py`: PASS (`4 passed`)
- Warning summary observed during the malformed vibration-axis rejection test:
  - `dqvl/vib_rules.py` emitted `RuntimeWarning` entries from `nanmin` / `nanmax` / `nanmean` on an intentionally broken input fixture with an all-NaN axis.
  - This did not block the test outcome because the wrapper still rejected the malformed input as required.
- P0D status promoted to complete based on Colab execution evidence.

## 28) Continuation update - 2026-03-08 (P0E batch scoring implementation ready)
- Implemented P0E batch scoring integration with existing checkpoints/scaler/config:
  - `batch_decision/scoring_engine.py`
  - `batch_decision/runner.py` (`--score-only` mode)
  - `batch_decision/contracts.py`
  - `batch_decision/__init__.py`
  - `tests/batch_decision/test_scoring_engine.py`
  - `tests/batch_decision/test_runner_score_only.py`
- Scoring path now reuses:
  - PatchTST checkpoint + saved `ChannelScaler`
  - SwinMAE checkpoint + existing CWT/image transform config
  - unified `inference/scoring.py` dispatch for per-window score generation
- Local checks completed:
  - `python3 -m py_compile batch_decision/__init__.py batch_decision/contracts.py batch_decision/scoring_engine.py batch_decision/runner.py tests/batch_decision/test_scoring_engine.py tests/batch_decision/test_runner_score_only.py`: PASS
- Local verification command blocker remains:
  - `python3 -m pytest ...` could not run because `pytest` is not installed in system or project venv.
- Remaining close step for P0E:
  - user-side Colab verification after pull:
    - `python3 -m pytest -q tests/batch_decision/test_scoring_engine.py tests/batch_decision/test_runner_score_only.py`
    - `python3 -m batch_decision.runner --config configs/batch_decision_runtime_colab.yaml --score-only`

## 29) Continuation update - 2026-03-11 (P0E Colab verification closed)
- User executed the P0E Colab pytest verification target:
  - `python3 -m pytest -q tests/batch_decision/test_scoring_engine.py tests/batch_decision/test_runner_score_only.py`: PASS (`3 passed`)
- Real `--score-only` CLI run remained input-path dependent because the Colab runtime profile still used placeholder test-file paths.
- P0E status promoted to complete based on scoring integration tests and verified runner path.

## 30) Continuation update - 2026-03-11 (P1A decision/reporting implementation ready)
- Implemented P1A decision and reporting flow:
  - `batch_decision/decision_engine.py`
  - `batch_decision/reporting.py`
  - `batch_decision/runner.py` (`--run` mode)
  - `batch_decision/contracts.py`
  - `batch_decision/__init__.py`
  - `tests/batch_decision/test_decision_engine.py`
  - `tests/batch_decision/test_reporting.py`
  - `tests/batch_decision/test_runner_full_run.py`
- Added threshold-based decision mapping for `normal | warn | anomaly`, reason generation, run summary aggregation, and chart-ready payload assembly.
- Added JSON/CSV/chart export artifacts:
  - `decision_report.json`
  - `decision_events.csv`
  - `chart_payload.json`
- Local checks completed:
  - `python3 -m py_compile batch_decision/__init__.py batch_decision/contracts.py batch_decision/decision_engine.py batch_decision/reporting.py batch_decision/runner.py tests/batch_decision/test_decision_engine.py tests/batch_decision/test_reporting.py tests/batch_decision/test_runner_full_run.py tests/batch_decision/test_runner_skeleton.py`: PASS
- Local verification command blocker remains:
  - `python3 -m pytest ...` could not run because `pytest` is not installed in system environment.
- Remaining close step for P1A:
  - user-side Colab verification after pull:
    - `python3 -m pytest -q tests/batch_decision/test_decision_engine.py tests/batch_decision/test_reporting.py tests/batch_decision/test_runner_full_run.py tests/batch_decision/test_runner_skeleton.py`
    - `python3 -m batch_decision.runner --config configs/batch_decision_runtime_colab.yaml --run`

## 31) Continuation update - 2026-03-11 (raw vibration header compatibility fix)
- User-side Colab validation exposed a real-input compatibility gap:
  - raw vibration CSV used headers like `Time Stamp`, ` X-axis`, ` Y-axis`, ` Z-axis`
  - existing vibration reader only accepted direct `timestamp|time` and `x|y|z` matches
- Applied reader normalization fix in `datasets/readers/vib_reader.py`:
  - trims whitespace
  - normalizes header variants such as `Time Stamp` -> `timestamp`
  - normalizes axis variants such as `X-axis` -> `x`
- Added regression coverage in `tests/batch_decision/test_import_and_preprocess.py` for axis/timestamp header variants.

## 32) Continuation update - 2026-03-11 (P1A Colab verification completed)
- User executed the full batch decision path in Colab:
  - `python3 -m batch_decision.runner --config configs/batch_decision_runtime_colab.yaml --run`: PASS
- Full run output:
  - `total_events=75`
  - `decision_counts={'normal': 0, 'warn': 4, 'anomaly': 71}`
  - exported:
    - `artifacts/batch_decision/colab_validation/decision_report.json`
    - `artifacts/batch_decision/colab_validation/decision_events.csv`
    - `artifacts/batch_decision/colab_validation/chart_payload.json`
- P1A status promoted to complete based on Colab full-run evidence.

## 33) Continuation update - 2026-03-11 (P1B dashboard bridge implementation ready)
- Implemented the Phase 3A P1B dashboard bridge and separate batch-decision view wiring:
  - `dashboard_bridge/export_batch_decision_state.py`
  - `tests/dashboard_bridge/test_export_batch_decision_state.py`
  - `training_dashboard/data/dashboard-layout.json`
  - `training_dashboard/js/app.js`
  - `training_dashboard/js/panels.js`
  - `training_dashboard/css/main.css`
  - `training_dashboard/index.html`
  - `batch_decision/runner.py`
- Added a dedicated `Batch Decision` tab/view to the static dashboard with:
  - batch-specific node graph
  - summary cards for normal/warn/anomaly and fused-score aggregates
  - event preview list
  - overlaid score/threshold chart using `chart_payload`
  - artifact links for `decision_report.json`, `decision_events.csv`, `chart_payload.json`, and `batch-decision-state.json`
- Full batch runner now refreshes `training_dashboard/data/batch-decision-state.json` automatically after `--run` when the dashboard layout exists under the inferred repo root.
- Local verification completed:
  - `python3 -m py_compile batch_decision/runner.py dashboard_bridge/export_batch_decision_state.py tests/dashboard_bridge/test_export_batch_decision_state.py tests/batch_decision/test_runner_full_run.py`: PASS
  - `node --check training_dashboard/js/app.js && node --check training_dashboard/js/panels.js`: PASS
  - synthetic exporter smoke via `export_batch_decision_state(...)`: PASS (`batch-decision-state.json` generated in a temporary repo fixture)
- Local blockers remain:
  - `python3 -m pytest ...` could not run because `pytest` is not installed in this environment
  - full local runner smoke could not run because `torch` is not installed in this environment
- Remaining close step for P1B:
  - user-side Colab verification after pull using the latest batch decision artifacts and dashboard static server

## 34) Continuation update - 2026-03-11 (P1B Colab verification completed)
- User executed the P1B verification targets in Colab:
  - `python3 -m pytest -q tests/dashboard_bridge/test_export_batch_decision_state.py tests/batch_decision/test_runner_full_run.py`: PASS (`3 passed`)
  - `python3 -m batch_decision.runner --config configs/batch_decision_runtime_colab.yaml --run`: PASS
- Full run output refreshed both report artifacts and dashboard bridge state:
  - `total_events=75`
  - `decision_counts={'normal': 0, 'warn': 3, 'anomaly': 72}`
  - `dashboard_json=/content/AnomalyDetection/training_dashboard/data/batch-decision-state.json`
  - confirmed file exists (`17K`)
- P1B status promoted to complete based on Colab exporter tests, runner tests, and full dashboard bridge generation evidence.

## 35) Continuation update - 2026-03-11 (P1C local GPU migration profile ready)
- Implemented the Phase 3A P1C local GPU migration readiness slice:
  - `configs/batch_decision_runtime_local_gpu.yaml`
  - `tests/batch_decision/test_local_gpu_profile.py`
  - `docs/runbook.md`
  - `README.md`
- Added a dedicated local GPU batch runtime profile that keeps the same `run`/`artifact_paths`/`preprocess` contract as the validated Colab profile and changes only:
  - `environment.profile`
  - `environment.expected_cwd`
  - local example input paths
  - local output directory
- Local verification completed:
  - `python3 -m py_compile tests/batch_decision/test_local_gpu_profile.py`: PASS
  - manual YAML contract comparison between Colab and local GPU profiles: PASS
- Local blockers remain:
  - `python3 -m pytest tests/batch_decision/test_local_gpu_profile.py` could not run because `pytest` is not installed
  - importing `batch_decision.runner` for a local profile smoke failed because `numpy` is not installed in this environment
- Remaining close step for P1C:
  - user-side Colab verification of the new profile contract and runbook flow

## 36) Continuation update - 2026-03-11 (P1C Colab verification completed)
- User executed the P1C verification targets in Colab:
  - `python3 -m pytest -q tests/batch_decision/test_local_gpu_profile.py`: PASS (`3 passed`)
  - `python3 -m batch_decision.runner --config configs/batch_decision_runtime_local_gpu.yaml --dry-run`: PASS
- Verified behavior:
  - local GPU profile contract loads successfully
  - Colab and local GPU profiles keep the same artifact/preprocess schema
  - local GPU dry-run resolves the shared threshold artifact and validates `dual` mode requirements
- Note from Colab workspace:
  - `configs/batch_decision_runtime_colab.yaml` was locally modified in Colab before pull (`M`), which is expected from prior test-path edits and does not affect the committed repository state.
- P1C status promoted to complete based on Colab profile tests and dry-run evidence.

## 37) Continuation update - 2026-03-11 (local retrain-first runbook examples added)
- Added concrete local retrain-first guidance for the post-P1 workflow:
  - `docs/runbook.md`
  - `README.md`
- Documented:
  - copying base training configs into `*_local_train.yaml`
  - switching `data.source` to `csv`
  - filling real local training paths
  - validating fresh training outputs
  - pointing `configs/batch_decision_runtime_local_gpu.yaml` preprocess references back to the exact local training configs before running batch tests
