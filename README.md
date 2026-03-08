# Anomaly Detection Framework (Local PC Flow)

This repository copy is prepared for local execution without notebooks.
Put your train and test data into the default folders, run one Python command for training, then run one Python command for test scoring and result export.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
```

Optional CUDA check:

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"
```

## Default Folder Layout

- `data/local/train/fdc/`
- `data/local/train/vib/`
- `data/local/test/fdc/`
- `data/local/test/vib/`

Detailed format notes are in `data/local/README.md`.

## Default Commands

Train both streams and export dashboard state:

```bash
python run_local_train.py
```

Run test scoring and export JSON/CSV decision results:

```bash
python run_local_test.py
```

## Optional Stream-Specific Commands

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

## Outputs

Training produces:

- `checkpoints/patchtst_ssl.pt`
- `checkpoints/swinmae_ssl.pt`
- `artifacts/scaler_fdc.json`
- `artifacts/loss/*.csv`
- `artifacts/loss/*.png`
- `training_dashboard/data/dashboard-state.json`

Test execution produces:

- `artifacts/batch_decision/local_gpu/<run_id>/summary.json`
- `artifacts/batch_decision/local_gpu/<run_id>/patchtst_events.json`
- `artifacts/batch_decision/local_gpu/<run_id>/patchtst_events.csv`
- `artifacts/batch_decision/local_gpu/<run_id>/swinmae_events.json`
- `artifacts/batch_decision/local_gpu/<run_id>/swinmae_events.csv`

## Default Configs

- `configs/patchtst_ssl_local.yaml`
- `configs/swinmae_ssl_local.yaml`
- `configs/batch_decision_runtime_local_gpu.yaml`

Defaults assume CSV files in `data/local/...` and prefer CUDA automatically when available.

## Notes

- `python run_local_train.py` uses the existing trainer modules, not notebooks.
- `python run_local_test.py` uses the trained checkpoints, PatchTST scaler, and default thresholds from `artifacts/thresholds/batch_decision_thresholds.json`.
- If you need different paths, you can override them with CLI options shown in `docs/runbook.md`.
