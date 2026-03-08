# Runbook (Local PC)

## 1. Environment setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
```

Optional CUDA check:

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"
```

## 2. Place data files

Default folder layout:

- `data/local/train/fdc/`
- `data/local/train/vib/`
- `data/local/test/fdc/`
- `data/local/test/vib/`

Format expectations:

- FDC CSV/parquet: optional `timestamp` or `time`; all other columns are numeric features.
- Vibration CSV: columns `x`, `y`, `z`, optional `timestamp` or `time`.
- Vibration NPY: shape `(T, 3)`.

## 3. Run training

Default command:

```bash
python run_local_train.py
```

What it does:

- trains PatchTST with `configs/patchtst_ssl_local.yaml`
- trains SwinMAE with `configs/swinmae_ssl_local.yaml`
- runs one scoring example per stream
- validates output artifacts
- exports `training_dashboard/data/dashboard-state.json`

Key training outputs:

- `checkpoints/patchtst_ssl.pt`
- `checkpoints/swinmae_ssl.pt`
- `artifacts/scaler_fdc.json`
- `artifacts/loss/patchtst_loss_history.csv`
- `artifacts/loss/swinmae_loss_history.csv`

## 4. Run test scoring and export results

Default command:

```bash
python run_local_test.py
```

What it does:

- loads `configs/batch_decision_runtime_local_gpu.yaml`
- preprocesses test files with the same training-compatible configs
- scores windows with trained checkpoints
- applies `normal | warn | anomaly` threshold mapping
- exports JSON and CSV reports

Default report location:

- `artifacts/batch_decision/local_gpu/<run_id>/summary.json`
- `artifacts/batch_decision/local_gpu/<run_id>/patchtst_events.json`
- `artifacts/batch_decision/local_gpu/<run_id>/patchtst_events.csv`
- `artifacts/batch_decision/local_gpu/<run_id>/swinmae_events.json`
- `artifacts/batch_decision/local_gpu/<run_id>/swinmae_events.csv`

## 5. Single-stream execution

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

## 6. Useful overrides

Train with explicit data globs:

```bash
python run_local_train.py   --patch-data-path "data/local/train/fdc/*.csv"   --swin-data-path "data/local/train/vib/*.csv"
```

Test with explicit paths or custom output directory:

```bash
python run_local_test.py   --patch-test-path "data/local/test/fdc/*.csv"   --swin-test-path "data/local/test/vib/*.csv"   --output-dir "artifacts/batch_decision/manual_run"
```

Dry-run config validation:

```bash
python run_local_test.py --dry-run
```

## 7. Common failure points

Missing training data:

- `run_local_train.py` fails if no files match `data.path` in local configs.

Missing test data:

- `run_local_test.py` fails if no files match `run.input_paths.*`.

Missing artifacts before test:

- test flow requires:
  - `checkpoints/patchtst_ssl.pt` when PatchTST is enabled
  - `checkpoints/swinmae_ssl.pt` when SwinMAE is enabled
  - `artifacts/scaler_fdc.json` for PatchTST
  - `artifacts/thresholds/batch_decision_thresholds.json`

## 8. Dashboard

After training, you can open the static dashboard with:

```bash
python -m http.server 8765 --directory training_dashboard
```
