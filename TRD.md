# TRD (Technical Requirements / Design Doc) - Phase 1

## 0) Decision Lock (Applied)

- CWT backend is fixed to `pywt` in Phase 1.
- Default normalization is `robust`; config keeps easy switch to `zscore`.
- Tensor contracts are fixed as FDC `(B, T, C)` and vibration image `(B, 3, H, W)`.
- Scoring contract is unified as `infer_score(batch, model, stream) -> (score, aux)`.

## 1) Tech Stack (Colab-Friendly)

- Python 3.10+
- PyTorch, torchvision
- timm (Swin backbone)
- numpy, pandas
- pywt (PyWavelets) and/or scipy for CWT backend abstraction
- einops, tqdm
- tensorboard (or optional wandb)

## 2) Recommended Repository Structure

```text
anomaly_framework/
  configs/
    patchtst_ssl.yaml
    swinmae_ssl.yaml
  datasets/
    fdc_dataset.py
    vib_dataset.py
    transforms/
      cwt.py
      normalization.py
      windowing.py
  models/
    patchtst/
      patchtst_ssl.py
    swinmae/
      swinmae_ssl.py
  trainers/
    train_patchtst_ssl.py
    train_swinmae_ssl.py
    utils.py
  notebooks/
    colab_patchtst_ssl.ipynb
    colab_swinmae_ssl.ipynb
  README.md
  requirements.txt
```

## 3) Cross-Cutting Design Principles

- Config-first: all key parameters in YAML
- Device auto-selection: CUDA if available, otherwise CPU
- Reproducibility: seed + deterministic options
- Train-only normalization statistics to prevent leakage
- Streaming-friendly slicing via sliding windows on continuous series

## 4) PatchTST-SSL Design

### Data Contract

- Input: `float32` continuous series `(T, C)`, `C >= 50`
- Window config: `seq_len` and `seq_stride`

### Preprocessing

- Channel-wise normalization (`mean/std` or robust `median/IQR`)
- Persist scaler artifact for train/val/test consistency

### Model Skeleton

- Patchify with configurable `patch_len` and stride
- Random patch masking with configurable `mask_ratio`
- Transformer encoder + lightweight reconstruction head
- Loss: masked-patch MSE only

### Outputs

- Checkpoint at `checkpoints/patchtst_ssl.pt`
- Optional scoring hook: aggregated reconstruction error

## 5) SwinMAE-SSL Design

### Data Contract

- Input vibration stream: `(T, 3)` float32
- Default sampling: `fs=2000`
- Windowing by `win_sec` and `win_stride_sec`

### CWT Pipeline

- Morlet CWT over configurable frequency range (`freq_min`, `freq_max`, `n_freqs`)
- `abs` magnitude, optional `log1p`, normalization, resize
- Build `(3, H, W)` image from x/y/z scalograms

### Model Skeleton

- Patch-level random masking on image input
- Swin encoder (timm) + lightweight decoder
- Loss: masked-region MSE

### Outputs

- Checkpoint at `checkpoints/swinmae_ssl.pt`
- Optional scoring hook: masked MSE / pixel MSE

## 6) Training Entry Points

```bash
python -m trainers.train_patchtst_ssl --config configs/patchtst_ssl.yaml
python -m trainers.train_swinmae_ssl --config configs/swinmae_ssl.yaml
```

## 7) Validation Requirements

- Colab CPU: smoke run completes end-to-end
- Colab GPU: CUDA path runs without code changes
- Synthetic data: both losses show downward trend in short runs
