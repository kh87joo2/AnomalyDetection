# TRD - Local CUDA Migration Track

Version: v1.0
Date: 2026-02-23
Owner: local-cuda track

## 1. Technical Objective
Implement a local CUDA execution path that is independent from Colab-specific runtime assumptions while preserving the same model and pipeline behavior.

## 2. Current Architecture Baseline
- Trainers: `trainers/train_patchtst_ssl.py`, `trainers/train_swinmae_ssl.py`
- Inference: `inference/run_scoring_example.py`
- Validation: `pipelines/validate_training_outputs.py`
- Dashboard export: `pipelines/export_training_dashboard_state.py`
- Config loader: `core/config.py`

The runtime code already supports CUDA auto-selection via `torch.cuda.is_available()`.

## 3. Pathing and Workspace Rules
TR-01: Use repository-relative paths by default.
TR-02: Avoid absolute `/content/...` paths in all local execution assets.
TR-03: Preserve existing Colab notebooks as legacy references.
TR-04: Local variants must live under `notebooks/local_*.ipynb`.
TR-05: Local config variants must live under `configs/*_local.yaml`.

## 4. Config Strategy
### 4.1 Required local configs
- `configs/patchtst_ssl_local.yaml`
- `configs/swinmae_ssl_local.yaml`

### 4.2 Required fields
- `device.prefer_cuda: true`
- `device.amp: true` (or false fallback depending on GPU stability)
- `logging.log_dir: runs/<stream>_ssl`
- `logging.checkpoint_path: checkpoints/<stream>_ssl.pt`
- data paths under `data/` with glob patterns
- DQVL report paths under `artifacts/dqvl/`

### 4.3 Compatibility
Validation/export scripts should be invoked with explicit `--patch-config` and `--swin-config` when needed to avoid ambiguity.

## 5. Notebook Strategy
- Keep Colab notebooks unchanged:
  - `notebooks/colab_patchtst_ssl.ipynb`
  - `notebooks/colab_swinmae_ssl.ipynb`
- Add local notebooks:
  - `notebooks/local_patchtst_ssl.ipynb`
  - `notebooks/local_swinmae_ssl.ipynb`
- Remove Colab-only cells from local notebooks:
  - `google.colab` imports
  - `/content` hardcoded paths
  - Colab shell magics tied to runtime filesystem assumptions

## 6. Artifact Migration Contract
Expected artifact locations after migration:
- Checkpoints:
  - `checkpoints/patchtst_ssl.pt`
  - `checkpoints/swinmae_ssl.pt`
- Loss:
  - `artifacts/loss/patchtst_loss_history.csv`
  - `artifacts/loss/patchtst_loss_curve.png`
  - `artifacts/loss/swinmae_loss_history.csv`
  - `artifacts/loss/swinmae_loss_curve.png`
- Logs: `runs/patchtst_ssl/`, `runs/swinmae_ssl/`
- Dashboard state: `training_dashboard/data/dashboard-state.json`
- Run history: `training_dashboard/data/runs/*.json`

## 7. Validation and Verification
V-01: Environment verification
- Python version
- CUDA visibility (`torch.cuda.is_available()`)

V-02: Static quality checks
- `pytest -q`
- `ruff check .`

V-03: Runtime checks
- train both streams
- run scoring smoke for both streams
- run validation checklist
- export dashboard state with run history persistence

## 8. Rollback and Safety
- Original baseline docs remain intact (`PRD.md`, `TRD.md`, `Todo.md`).
- Lowercase routing links (`prd.md`, `trd.md`, `todo.md`) can be switched back to baseline files in one command.
- No deletion of legacy notebooks or historical docs.

## 9. Implementation Sequence
Step 1: Add local docs and route lowercase links to local docs.
Step 2: Add local config files and local notebooks.
Step 3: Import or regenerate required artifacts.
Step 4: Execute full local validation pipeline.
Step 5: Publish final local runbook and handoff checklist.
