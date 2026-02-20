# Todo (Phase 1.5 Execution Plan - Real-Data Connection) [2026-02-20]

## Document Scope Tags

- Legacy baseline (Phase 1): before 2026-02-20
- Current execution scope (Phase 1.5): 2026-02-20

## 0) Baseline Status (Completed)

1. Phase 1 synthetic pipelines are implemented for PatchTST-SSL and SwinMAE-SSL.
2. Existing trainer entry points run end-to-end in Colab.
3. Synthetic smoke training and scoring paths are validated.

## 1) Decision Lock for Phase 1.5

1. Keep existing trainer entry points unchanged.
2. Add real-data path only in dataset builders via `data.source` branching.
3. Reader layer is as-is load only (no auto sort/fix/impute).
4. DQVL-lite uses hard-fail and warning separation.
5. DQ decision is file-level only (`keep|drop`) for this phase.
6. Split first, window later; no cross-file windows.
7. Keep existing config keys; add only minimal new keys.

## 2) P0 - Contracts and Readers

1. Add `datasets/contracts.md` with FDC/vibration input schema and examples.
2. Add `datasets/readers/fdc_reader.py` for CSV/Parquet raw load.
3. Add `datasets/readers/vib_reader.py` for CSV/NPY raw load.
4. Ensure reader outputs numeric arrays as `float32` and returns metadata needed by downstream steps.

## 3) P0 - DQVL-lite

1. Add `dqvl/fdc_rules.py` and implement minimum hard-fail/warning checks.
2. Add `dqvl/vib_rules.py` and implement minimum hard-fail/warning checks.
3. Add `dqvl/report.py` and standardize JSON schema fields:
   - `schema_version`, `run_id`, `file_id`, `decision`, `hard_fails`, `warnings`, `metrics`
4. Add `dqvl.enabled=false` behavior with explicit `decision='skipped'` report output.

## 4) P0 - Dataset Builder Integration

1. Update `build_fdc_datasets(config)`:
   - support `data.source=synthetic|csv|parquet`
   - real-data flow: read -> dqvl -> split -> scaler fit(train) -> transform -> window
2. Update `build_vibration_datasets(config)`:
   - support `data.source=synthetic|csv|npy`
   - real-data flow: read -> dqvl -> split -> optional configured resample -> window -> CWT
3. Enforce no cross-file window generation.

## 5) P0 - Config Updates (Backward Compatible)

1. Add minimal keys to configs without breaking current keys:
   - `data.source`, `data.path`
   - `dqvl.enabled`, `dqvl.allow_sort_fix`, `dqvl.hard_fail.*`, `dqvl.warn.*`
   - vibration only: `data.resample.enabled`, `data.resample.method`
2. Keep existing keys currently used by trainers/builders untouched.

## 6) P0 - Tests and Smoke Validation

1. Add dummy files:
   - `tests/smoke/data/fdc_dummy.csv`
   - `tests/smoke/data/vib_dummy.csv`
2. Add tests:
   - `tests/test_fdc_csv_smoke.py`
   - `tests/test_vib_csv_smoke.py`
3. Validate one-batch forward + finite loss for both real-data paths.
4. Run regression check for existing synthetic tests.

## 7) P1 - Deferred (Not in Phase 1.5)

1. New pipeline entry points under `pipelines/`.
2. Advanced DQ quality scoring and weighting strategy.
3. Automatic fs estimation and adaptive resampling.
4. Large-scale data optimization (chunking/memmap/distributed).
5. Aggregated dashboard/reporting UI.

## 8) Exit Criteria

1. Existing trainer commands run with real-data config and synthetic config.
2. DQVL report JSON is generated per input file.
3. No split/window leakage violations.
4. Real-data smoke tests pass.
