# Real-Data Contracts (Phase 1.5)

## FDC Stream (PatchTST)
- Input format: `.csv` or `.parquet`
- Required column: `timestamp` (case-insensitive: `timestamp`, `time`)
- Feature columns: all non-timestamp columns, numeric-castable
- Reader behavior: raw row order is preserved (`as-is`), no sorting/interpolation
- Model input: window tensor `(N, T, C)`

Example CSV header:

```csv
timestamp,param_01,param_02,param_03
2026-01-01T00:00:00,1.0,2.0,3.0
```

## Vibration Stream (SwinMAE)
- Input format: `.csv` or `.npy`
- CSV required axis columns: `x`, `y`, `z` (case-insensitive)
- Optional CSV timestamp column: `timestamp` or `time`
- NPY shape must be `(T, 3)`
- Reader behavior: raw order is preserved (`as-is`), no interpolation/resample
- Model input: CWT image tensor `(N, 3, H, W)` after windowing and transform

Example CSV header:

```csv
timestamp,x,y,z
2026-01-01T00:00:00.000,0.01,-0.02,0.03
```

## DQVL-lite Decision Contract
- Output unit: `file_id`
- Decision: `keep` or `drop`
- Report fields:
  - `schema_version`
  - `run_id`
  - `file_id`
  - `decision`
  - `hard_fails`
  - `warnings`
  - `metrics`
