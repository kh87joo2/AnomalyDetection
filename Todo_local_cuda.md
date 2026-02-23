# Todo - Local CUDA Migration Track

Version: v1.0
Date: 2026-02-23

## 1) Rules
- Keep baseline docs/code available for reference.
- Implement migration work in this local-cuda track only.
- Prefer repo-relative paths and deterministic commands.

## 2) P0 - Planning and Routing
- [x] T01: Create and approve local-cuda PRD/TRD/Todo documents.
- [x] T02: Route `prd.md`, `trd.md`, `todo.md` to local-cuda docs for Ralph loop discovery.

Definition of done:
- New local docs exist and are readable.
- Lowercase routing links point to local docs.

## 3) P0 - Local Environment Bootstrap
- [ ] T03: Create clean local `.venv` and install runtime/dev requirements.
- [ ] T04: Verify CUDA runtime visibility (`torch.cuda.is_available()` and device name).

Definition of done:
- Dependencies installed without conflict.
- CUDA torch check output recorded.

## 4) P0 - Config Split
- [x] T05: Create `configs/patchtst_ssl_local.yaml`.
- [x] T06: Create `configs/swinmae_ssl_local.yaml`.
- [x] T07: Ensure all data/log/checkpoint/report paths are local-relative.
- [x] T07A: Add runtime config override flow for local Kaggle paths (`--patch-data-path`, `--swin-data-path`).

Definition of done:
- Both local configs parse.
- No `/content` path in local configs.

## 5) P0 - Notebook Split
- [x] T08A: Add notebook-free Python integration runner (`pipelines/run_local_training_pipeline.py`).
- [ ] T08: Add `notebooks/local_patchtst_ssl.ipynb`.
- [ ] T09: Add `notebooks/local_swinmae_ssl.ipynb`.
- [ ] T10: Remove Colab-only dependencies and hardcoded runtime paths from local notebooks.

Definition of done:
- Local notebooks execute from repository root.
- Colab notebooks remain unchanged.

## 6) P0 - Artifact Recovery and Validation
- [ ] T11: Import existing Colab artifacts (if available) into local standard directories.
- [ ] T12: Run PatchTST training with local config.
- [ ] T13: Run SwinMAE training with local config.
- [ ] T14: Run scoring smoke for both streams.
- [ ] T15: Run `pipelines.validate_training_outputs` with local config references.

Definition of done:
- Required checkpoints/loss/log/scaler artifacts exist.
- Validation command exits successfully or documented known blockers exist.

## 7) P0 - Dashboard Integration
- [x] T16A: Add dashboard API server + upload/train controls (`training_dashboard/server.py`, UI buttons).
- [x] T16B: Show live running stage on node graph (active node glow from `/api/status` live node map).
- [ ] T16: Export dashboard state JSON from local run.
- [ ] T17: Persist run history snapshot and index.
- [ ] T18: Launch local dashboard static server and confirm render.

Definition of done:
- `training_dashboard/data/dashboard-state.json` is generated.
- `training_dashboard/data/runs/index.json` and run snapshot exist.

## 8) P1 - Hardening
- [ ] T19: Add one-click local bootstrap helper script.
- [x] T20: Update `docs/runbook.md` with local-first execution flow.
- [ ] T21: Add troubleshooting section for CUDA and path issues.

## 9) Verification Commands
- `pytest -q`
- `ruff check .`
- `python -m trainers.train_patchtst_ssl --config configs/patchtst_ssl_local.yaml`
- `python -m trainers.train_swinmae_ssl --config configs/swinmae_ssl_local.yaml`
- `python -m inference.run_scoring_example --stream patchtst --checkpoint checkpoints/patchtst_ssl.pt --config configs/patchtst_ssl_local.yaml`
- `python -m inference.run_scoring_example --stream swinmae --checkpoint checkpoints/swinmae_ssl.pt --config configs/swinmae_ssl_local.yaml`
- `python -m pipelines.validate_training_outputs --repo-root . --patch-config configs/patchtst_ssl_local.yaml --swin-config configs/swinmae_ssl_local.yaml`
- `python -m pipelines.export_training_dashboard_state --repo-root . --out training_dashboard/data/dashboard-state.json --persist-run-history --run-id local_run_001`
- `python3 -m pipelines.run_local_training_pipeline --repo-root . --patch-config configs/patchtst_ssl_local.yaml --swin-config configs/swinmae_ssl_local.yaml --patch-data-source csv --patch-data-path "/absolute/path/to/patchtst_data/*.csv" --swin-data-source csv --swin-data-path "/absolute/path/to/swinmae_data/**/*.csv" --persist-run-history --validate-skip-smoke`
- `python3 -m training_dashboard.server --host 127.0.0.1 --port 8765`

## 10) Worklog
- 2026-02-23: `docs/local_cuda_worklog_2026-02-23.md`
