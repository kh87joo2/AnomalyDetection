# Phase 1 Contracts (Frozen)

## Environment Contract
- Runtime targets:
  - Validation now: Google Colab (CPU and CUDA GPU)
  - Deployment later: local CUDA GPU PC
- Device behavior: auto-select CUDA when available, otherwise CPU.

## Data Contracts

### PatchTST Stream (FDC)
- Raw series shape: `(T, C)` float32.
- Batch shape into model: `(B, T, C)`.
- Normalization default: `robust` (`median`/`IQR`), with `zscore` as easy config switch.

### SwinMAE Stream (Vibration)
- Raw series shape: `(T, 3)` float32.
- Windowed vibration shape: `(win_len, 3)`.
- Model input shape: `(B, 3, H, W)` from Morlet CWT (`pywt`) per axis.

## Training Objective Contract
- PatchTST: masked reconstruction (MSE on masked patches only).
- SwinMAE: masked reconstruction (MSE on masked image regions only).

## Scoring API Contract
- Standard function:
  - `infer_score(batch, model, stream) -> (score, aux)`
- Output:
  - `score`: float tensor shape `(B,)`
  - `aux`: stream-specific diagnostics (mask coverage, per-channel/axis errors)

## Checkpoint Contract
- Save path default: `checkpoints/<stream>_ssl.pt`
- Minimal required fields:
  - `model_state_dict`
  - `optimizer_state_dict` (train-time checkpoint)
  - `epoch`
  - `config`
