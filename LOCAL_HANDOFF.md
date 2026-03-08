# Local Handoff

## 1. Summary

This copied workspace is prepared for local PC execution without notebooks.
The original source folder was not modified.
All local-first changes were applied in this folder:

- `run_local_train.py`: one-command local training entrypoint
- `run_local_test.py`: one-command local test/result entrypoint
- `configs/patchtst_ssl_local.yaml`
- `configs/swinmae_ssl_local.yaml`
- `configs/batch_decision_runtime_local_gpu.yaml`
- `data/local/README.md`
- updated `README.md`
- updated `docs/runbook.md`

## 2. What Changed

### Local training flow

`python run_local_train.py`

This command runs the notebook-free local training pipeline and:

- trains PatchTST
- trains SwinMAE
- runs a scoring smoke step for each stream
- validates output artifacts
- exports dashboard state

### Local test flow

`python run_local_test.py`

This command:

- loads the local batch runtime config
- preprocesses local test files
- scores windows using trained checkpoints
- applies `normal | warn | anomaly` thresholds
- exports JSON and CSV result files

### Local data layout

Default folders:

- `data/local/train/fdc/`
- `data/local/train/vib/`
- `data/local/test/fdc/`
- `data/local/test/vib/`

## 3. Usage

### Environment setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
```

Optional CUDA check:

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"
```

### Train

Put training files into:

- `data/local/train/fdc/`
- `data/local/train/vib/`

Run:

```bash
python run_local_train.py
```

### Test

Put test files into:

- `data/local/test/fdc/`
- `data/local/test/vib/`

Run:

```bash
python run_local_test.py
```

### Single-stream examples

PatchTST only:

```bash
python run_local_train.py --skip-swinmae
python run_local_test.py --stream patchtst
```

SwinMAE only:

```bash
python run_local_train.py --skip-patchtst
python run_local_test.py --stream swinmae
```

## 4. Expected Data Format

### FDC

- default local config expects CSV files
- parquet is also supported if config is changed
- optional timestamp column: `timestamp` or `time`
- every non-timestamp column is treated as a numeric feature channel

### Vibration

- default local config expects CSV files
- NPY is also supported if config is changed
- CSV columns must include `x`, `y`, `z`
- optional timestamp column: `timestamp` or `time`
- NPY shape must be `(T, 3)`

## 5. Output Files

### After training

- `checkpoints/patchtst_ssl.pt`
- `checkpoints/swinmae_ssl.pt`
- `artifacts/scaler_fdc.json`
- `artifacts/loss/patchtst_loss_history.csv`
- `artifacts/loss/swinmae_loss_history.csv`
- `training_dashboard/data/dashboard-state.json`

### After test run

Results are written under a run-specific folder:

- `artifacts/batch_decision/local_gpu/<run_id>/summary.json`
- `artifacts/batch_decision/local_gpu/<run_id>/patchtst_events.json`
- `artifacts/batch_decision/local_gpu/<run_id>/patchtst_events.csv`
- `artifacts/batch_decision/local_gpu/<run_id>/swinmae_events.json`
- `artifacts/batch_decision/local_gpu/<run_id>/swinmae_events.csv`

## 6. Verification Completed

Verified in this copied workspace:

- `tests/pipelines/test_run_local_training_pipeline.py`
- `tests/pipelines/test_run_local_test_pipeline.py`

Result:

- `8 passed`

Dry-run verification also passed:

```bash
python run_local_train.py --dry-run --patch-data-path tests/smoke/data/fdc_dummy.csv --swin-data-path tests/smoke/data/vib_dummy.csv
python run_local_test.py --dry-run --stream patchtst --patch-test-path tests/smoke/data/fdc_dummy.csv
```

## 7. Git Setup

This copied folder was created without the original `.git` directory.
That means it should be treated as a separate repository.

Recommended local steps:

```bash
git init -b main
git add .
git commit -m "feat(local): add local-first train and test workflow"
```

If you later create a new remote repository on GitHub:

```bash
git remote add origin <YOUR_NEW_REPO_URL>
git push -u origin main
```

## 8. Recommended Files To Check First

- `LOCAL_HANDOFF.md`
- `README.md`
- `docs/runbook.md`
- `data/local/README.md`
- `run_local_train.py`
- `run_local_test.py`
