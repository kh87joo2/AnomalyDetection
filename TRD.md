# TRD (Technical Requirements / Design Doc) - Phase 1.5 Real-Data Connection

## 0) Decision Lock (Applied)

- Keep existing trainer entry points unchanged.
- Add real-data support only through dataset builders.
- Reader must load data as-is (no auto-fix).
- DQVL-lite must output hard-fail and warning separately.
- DQ decision unit is file-level in this phase.
- Split first, window later; no cross-file windows.
- Existing config keys remain valid; add only minimal new keys.

## 1) Target Repository Additions

```text
configs/
  patchtst_ssl.yaml          # keep existing keys + minimal additions
  swinmae_ssl.yaml           # keep existing keys + minimal additions

datasets/
  contracts.md
  readers/
    fdc_reader.py
    vib_reader.py

dqvl/
  report.py
  fdc_rules.py
  vib_rules.py

tests/
  smoke/data/
    fdc_dummy.csv
    vib_dummy.csv
  test_fdc_csv_smoke.py
  test_vib_csv_smoke.py
```

## 2) Data Contracts

### A) FDC Contract

- Input: CSV/Parquet
- Required fields:
  - one timestamp column (configured)
  - N process channels (`N >= 1`, typically large)
- Target tensor contract into model path: `(B, T, C)`

### B) Vibration Contract

- Input: CSV/NPY
- Required fields:
  - x/y/z axes
  - timestamp or sample index
  - fs from config or metadata
- Target model input contract: `(B, 3, H, W)` after window + CWT

## 3) Config Contract (Backward Compatible)

### Existing Keys

- Keep all current keys used by trainers/builders unchanged.
- Do not migrate existing keys to new nesting in this phase.

### Minimal Added Keys

- Common:
  - `data.source`: `synthetic|csv|parquet|npy` (stream-specific allowed values)
  - `data.path`: input file path or glob
- DQVL:
  - `dqvl.enabled`: `true|false`
  - `dqvl.allow_sort_fix`: `false|true` (default `false`)
  - `dqvl.hard_fail.*`: threshold set
  - `dqvl.warn.*`: threshold set
- Vibration:
  - `data.resample.enabled`: `false|true`
  - `data.resample.method`: fixed deterministic method for this phase

## 4) Reader Behavior

### A) FDC Reader (`datasets/readers/fdc_reader.py`)

- Responsibilities:
  - Load raw table from CSV/Parquet
  - Select configured timestamp and channel columns
  - Convert numeric channels to `float32`
  - Return raw order data and metadata
- Non-responsibilities:
  - No sorting
  - No dedup
  - No fill/impute
  - No outlier clipping

### B) Vibration Reader (`datasets/readers/vib_reader.py`)

- Responsibilities:
  - Load CSV/NPY to `(T, 3)` float32 + metadata
  - Read fs from config/metadata source by policy
- Non-responsibilities:
  - No implicit resample
  - No implicit signal repair

## 5) DQVL-lite Rules

### A) FDC Hard Fail (drop)

- Missing required columns
- Timestamp severe invalidity (non-monotonic or duplicate beyond threshold)
- Global missing ratio above hard threshold

### B) FDC Warning (keep with warning)

- Missing ratio above warning threshold
- Stuck/near-constant channel detected
- Step-jump ratio above warning threshold

### C) Vibration Hard Fail (drop)

- Missing required axes (`x,y,z`) or required metadata
- NaN ratio above hard threshold
- FS mismatch while `data.resample.enabled=false`

### D) Vibration Warning (keep with warning)

- Clipping ratio above warning threshold
- Flat-line ratio above warning threshold
- FS mismatch with `data.resample.enabled=true` before configured resample

## 6) DQVL Report Contract

- Required JSON fields:
  - `schema_version` (string)
  - `run_id` (string/uuid)
  - `file_id` (string)
  - `decision` (`keep|drop|skipped`)
  - `hard_fails` (list[str])
  - `warnings` (list[str])
  - `metrics` (object)
- `dqvl.enabled=false` policy:
  - `decision='skipped'` and record reason in report.

## 7) Split and Leakage Policy

- Split order is fixed:
  - read -> dqvl -> split by time -> windowing
- No windows may span across different files.
- Default file policy:
  - treat files independently for split/window generation
  - no implicit cross-file concatenation

## 8) FS Mismatch and Resampling Policy

- If configured fs and observed fs mismatch:
  - `resample.enabled=false` -> hard fail/drop
  - `resample.enabled=true` -> resample by configured deterministic method only
- No automatic fs estimation-based adaptive behavior in this phase.

## 9) Dataset Builder Integration

### A) PatchTST Path

- `build_fdc_datasets(config)`:
  - branch on `data.source`
  - synthetic path unchanged
  - real-data path: reader -> dqvl -> split -> scaler fit(train) -> transform -> window

### B) SwinMAE Path

- `build_vibration_datasets(config)`:
  - branch on `data.source`
  - synthetic path unchanged
  - real-data path: reader -> dqvl -> split -> (optional configured resample) -> window -> CWT image

## 10) Validation and Testing

- Keep existing smoke tests for synthetic path.
- Add real-data smoke tests:
  - `test_fdc_csv_smoke.py`
  - `test_vib_csv_smoke.py`
- Each test validates:
  - loader path works
  - one batch forward works
  - loss is finite

## 11) Acceptance Criteria

- Same trainer CLI commands work for synthetic and real-data modes.
- DQVL report generated for real-data ingestion path.
- No leakage from split/window policy violations.
- Real-data smoke tests pass.
