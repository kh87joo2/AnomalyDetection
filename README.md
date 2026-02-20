# Anomaly Detection Framework (Phase 1 Skeleton)

This repository contains the first runnable core for two self-supervised anomaly modeling streams:

- PatchTST-SSL for multivariate FDC time series
- SwinMAE-SSL for vibration (x, y, z) transformed by Morlet CWT (`pywt`)

## Fixed decisions applied

- Code root: current repository folder
- Objective: masked reconstruction for both streams
- CWT backend: `pywt` (fixed in Phase 1)
- Default normalization: `robust` (easy switch to `zscore` in config)
- Runtime strategy: validate in Google Colab with CUDA GPU, then migrate unchanged commands to local CUDA PC

## Repository structure

- `configs/`: training configs
- `datasets/`: synthetic generators, dataset builders, transforms
- `models/`: PatchTST and SwinMAE skeleton models
- `trainers/`: training entry points and shared utils
- `inference/`: unified score API and adapters
- `core/`: config loader and contracts

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Testing

Smoke tests were added for quick local verification.

```bash
pytest -q
```

## Train (smoke validation)

PatchTST-SSL:

```bash
python -m trainers.train_patchtst_ssl --config configs/patchtst_ssl.yaml
```

SwinMAE-SSL:

```bash
python -m trainers.train_swinmae_ssl --config configs/swinmae_ssl.yaml
```

## Colab GPU validation flow

1. Open Colab and set runtime to GPU.
2. Clone/upload this repository.
3. Install requirements.
4. Run both training commands above.
5. Verify checkpoint outputs:
- `checkpoints/patchtst_ssl.pt`
- `checkpoints/swinmae_ssl.pt`

## Notebooks and scoring example

Colab notebooks:
- `notebooks/colab_patchtst_ssl.ipynb`
- `notebooks/colab_swinmae_ssl.ipynb`

Scoring example:
```bash
python -m inference.run_scoring_example --stream patchtst --checkpoint checkpoints/patchtst_ssl.pt --config configs/patchtst_ssl.yaml
python -m inference.run_scoring_example --stream swinmae --checkpoint checkpoints/swinmae_ssl.pt --config configs/swinmae_ssl.yaml
```

## Unified scoring API

Use `inference/scoring.py`:

```python
from inference.scoring import infer_score

out = infer_score(batch, model, stream="patchtst")
score = out["score"]     # (B,)
aux = out["aux"]         # diagnostics
```

## Local CUDA PC migration

Use the same config and command lines. Only ensure CUDA-compatible PyTorch is installed on the target PC.
