# Calibration Split Policy

This document fixes the data split boundary used after training for threshold calibration.

## Goal

- Keep model training data and threshold calibration data separated.
- Use only normal windows for calibration.

## Split Criteria

- Train set:
  - Source: system-specific normal period used for model fitting.
  - Rule: do not reuse these windows for threshold estimation.
- Calibration set (normal only):
  - Source: later time period or separate files not used in training.
  - Rule: use this set to estimate score distributions and final threshold.

## Example (edit per system)

- Train period: `2026-01-01 00:00:00` to `2026-01-20 23:59:59`
- Calibration period (normal): `2026-01-21 00:00:00` to `2026-01-25 23:59:59`

## Notes

- PatchTST and SwinMAE thresholds should be estimated on aligned calibration windows.
- Final fused threshold must be derived from fused score distribution, not training loss curves.

