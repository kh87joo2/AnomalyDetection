# Todo (Legacy Phase 1)

Source commit: `3835f65`

# Todo (Phase 1 Execution Plan)

## Decision Lock (Applied)

1. Current repository folder is the code root.
2. PatchTST and SwinMAE both use masked reconstruction objective.
3. CWT backend is fixed to `pywt`.
4. Default normalization is `robust` with easy future switch to `zscore`.
5. Validate first on Colab CUDA and migrate commands unchanged to local CUDA PC.

## P0 - Running Skeleton First

1. Create repo structure and `requirements.txt` for Colab-first execution.
2. Build shared training utilities for seed control, auto device selection (`cpu/cuda`) with AMP option, checkpoint save/load, and TensorBoard logging.
3. Add YAML config loader and validation.

## P0 - PatchTST-SSL Stream

1. Implement synthetic FDC generator with shape `(T, C)` (default `C=64`), baseline patterns/noise, and optional anomaly injection hooks.
2. Implement FDC dataset windowing from continuous stream to sliding windows with a time-safe train/val split.
3. Implement FDC normalizer that fits on train split only and persists/loads scaler artifacts.
4. Implement patchify + masking module.
5. Implement minimal PatchTST-SSL model (encoder + reconstruction head) with masked MSE objective.
6. Add training script and smoke test with 1 epoch run, loss logging, and checkpoint output.

## P0 - SwinMAE-SSL Stream

1. Implement synthetic vibration generator for continuous `(T, 3)` data compatible with `fs=2000`.
2. Implement vibration dataset windowing to sample `(4000, 3)` windows by stride.
3. Implement CWT transform module with Morlet CWT backend selection (`pywt/scipy`), frequency controls, magnitude/log options, normalization, and resize.
4. Implement minimal SwinMAE-SSL model with patch masking, Swin encoder, lightweight decoder, and masked MSE objective.
5. Add training script and smoke test with checkpoint save and optional reconstruction sample dump.

## P1 - Validation Convenience and Portability

1. Create two Colab notebooks (PatchTST-SSL, SwinMAE-SSL).
2. Write README runbook covering install steps, training commands, config examples, and real-data replacement guide.
3. Standardize optional anomaly scoring API: `infer_score(batch) -> score, aux`.
