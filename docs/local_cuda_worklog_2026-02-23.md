# Local CUDA Worklog (2026-02-23)

Date: 2026-02-23  
Branch: `local-cuda`  
Workspace: `/home/userkh/workspace/ML_test/anomalydetection_local_cuda`

## 1) What Was Completed

### A. Workspace isolation and planning docs
- Created local migration documents:
  - `PRD_local_cuda.md`
  - `TRD_local_cuda.md`
  - `Todo_local_cuda.md`
- Routed lowercase planning links to local track docs:
  - `prd.md -> PRD_local_cuda.md`
  - `trd.md -> TRD_local_cuda.md`
  - `todo.md -> Todo_local_cuda.md`

### B. Local-first config and notebook-free execution path
- Added local config variants:
  - `configs/patchtst_ssl_local.yaml`
  - `configs/swinmae_ssl_local.yaml`
- Added integrated local runner:
  - `pipelines/run_local_training_pipeline.py`
- Implemented runtime override flow for local dataset paths:
  - `--patch-data-source`, `--patch-data-path`
  - `--swin-data-source`, `--swin-data-path`
- Runtime configs are generated under:
  - `artifacts/runtime_configs/`

### C. Dashboard control flow for local training
- Added local dashboard backend package init:
  - `training_dashboard/__init__.py`
- Added local dashboard API server:
  - `training_dashboard/server.py`
- Added UI controls for:
  - PatchTST file import
  - SwinMAE file import
  - `Train`, `Refresh`, `Stop`
- Added upload storage handling under:
  - `training_dashboard/uploads/` (ignored by git)

### D. Live stage visualization on node graph
- Added server-side live progress mapping from pipeline logs:
  - step parsing: `[i/n] step_name`
  - active step metadata: `active_step`, `step_index`, `step_total`
  - node status map: `live_nodes`
- Added frontend live-node overlay:
  - `live_nodes` from `/api/status` are merged with runtime nodes
  - node/edge state updates during run: `idle/running/done/fail`
- Result:
  - After pressing `Train`, currently running pipeline stage is highlighted in real time.

### E. Tests and documentation updates
- Added pipeline runner tests:
  - `tests/pipelines/test_run_local_training_pipeline.py`
- Added dashboard live progress tests:
  - `tests/training_dashboard/test_server_progress.py`
- Updated operator docs:
  - `docs/runbook.md`
  - `notebooks/README.txt`
- Updated dashboard frontend assets:
  - `training_dashboard/index.html`
  - `training_dashboard/js/app.js`
  - `training_dashboard/css/main.css`

## 2) Verification History (Executed in this track)
- `python3 -m pipelines.run_local_training_pipeline ... --dry-run`  
  - Confirmed step order and runtime config generation path.
- `python3 -m py_compile training_dashboard/server.py pipelines/run_local_training_pipeline.py`  
  - Passed.
- `node --check training_dashboard/js/app.js`  
  - Passed.
- TrainingJob progress simulation (manual Python snippet)  
  - Confirmed step-to-node transition and fail/success state propagation.

Note:
- Full `pytest` execution is pending because `pytest` is not installed in current environment yet.

## 3) Next Work (Recommended Order)

### P0 (must do next)
1. Environment bootstrap
   - Create local `.venv`
   - Install requirements/dev requirements
   - Verify CUDA visibility (`torch.cuda.is_available()`, device name)
2. Local artifact generation
   - Run PatchTST local training
   - Run SwinMAE local training
   - Run scoring smoke for both streams
   - Run `pipelines.validate_training_outputs`
3. Dashboard output finalization
   - Generate `training_dashboard/data/dashboard-state.json`
   - Persist `training_dashboard/data/runs/index.json` and run snapshots
   - Validate dashboard render end-to-end with real local run outputs

### P1 (after P0 stable)
1. Add one-click bootstrap helper script
2. Add CUDA/path troubleshooting section
3. Add dashboard-side hyperparameter override UI + runtime config merge flow

## 4) Commit Scope Intent
- This worklog records all local-cuda track changes up to 2026-02-23 before first branch commit.
- Push is intentionally deferred to the user.
