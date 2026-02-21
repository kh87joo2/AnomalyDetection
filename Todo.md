# Todo (Phase 1.5 Execution Plan - Real-Data Connection) [2026-02-20]

## Document Scope Tags

- Legacy baseline (Phase 1): before 2026-02-20
- Current execution scope (Phase 1.5): 2026-02-20
- Legacy file: `docs/archive/phase1_legacy/Todo_phase1.md`

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

## 9) Progress Snapshot (2026-02-20 EOD)

### Completed today
1. Implemented readers, DQVL-lite, dataset source branching, and config extension.
2. Added real-data smoke fixtures/tests (`tests/test_fdc_csv_smoke.py`, `tests/test_vib_csv_smoke.py`).
3. Updated both Colab notebooks with:
   - Kaggle download/prep cells.
   - Data check cells.
   - Training cells preferring `*_real.yaml`.
4. Committed and pushed changes:
   - `8ee3af8 feat(phase1.5): real-data pipeline + colab data flow [2026-02-20]`

### Immediate next execution tasks (resume point)
1. Execute `notebooks/colab_patchtst_ssl.ipynb` sequentially and confirm checkpoint creation.
2. Execute `notebooks/colab_swinmae_ssl.ipynb` sequentially.
3. In SwinMAE notebook, set `configs/swinmae_ssl_real.yaml:data.fs` from data check cell estimate before training.
4. Confirm DQVL JSON outputs exist and inspect `decision`, `hard_fails`, `warnings`.
5. Run inference scoring smoke for both streams with generated real-data checkpoints/configs.

## 10) Progress Snapshot (2026-02-21 EOD)

### Completed today
1. Ran both Colab notebooks through full execution path with real-data flow.
2. Verified PatchTST training logs with epoch-level loss outputs.
3. Completed SwinMAE data checks and real-data training flow.
4. Set `configs/swinmae_ssl_real.yaml` `data.fs` to `10000` after timestamp-estimation anomaly investigation.
5. Updated notebook training cells to stream logs in real time for clearer progress tracking.
6. Committed and pushed notebook refinements:
   - `405affb chore(notebooks): refine colab execution flow [2026-02-21]`

### Current blocker
1. Colab GPU quota/runtime instability prevented stable continuation.

### Immediate next execution tasks (resume point)
1. Recover stable Colab runtime and confirm CUDA availability.
2. Re-run minimal required cells only:
   - bootstrap
   - Kaggle/data-check cells
   - training cells
   - checkpoint check cells
3. Run scoring smoke on both streams using real configs.
4. Verify DQVL reports under `artifacts/dqvl/fdc` and `artifacts/dqvl/vib`.
5. Save final artifacts bundle to persistent storage.

### Fallback path (if GPU quota still exhausted)
1. Lower epochs and max batches for CPU smoke verification only.
2. Defer full real-data training until GPU quota reset.

## 11) Progress Snapshot (2026-02-21 Late Session)

### Completed today (additional)
1. Diagnosed PatchTST real-data loss explosion using runtime scale diagnostics:
   - observed `abs_p99 ~= 31057`
   - observed tiny robust scales (`min ~= 0.000128`, `tiny_scale_count(<0.05)=9`)
2. Built stabilized PatchTST input file:
   - `/content/AnomalyDetection/data/fdc/swat_fdc_train_clean_v2.csv`
   - channel-preserving low-variance handling + outlier clipping
3. Updated PatchTST real config for stable training:
   - `data.path` -> `swat_fdc_train_clean_v2.csv`
   - `training.lr = 1e-4`
   - `device.amp = false`
   - `training.epochs = 10`
4. Completed PatchTST 10-epoch retraining with stable decreasing trend:
   - Epoch 1: `train=8.029775`, `val=39.985680`
   - Epoch 10: `train=4.214598`, `val=21.153209`
5. Final artifacts confirmed and bundled:
   - `/content/run_bundle_20260221_095519`
   - contains: both checkpoints + both real configs

### Next actions (Phase 2 kickoff)
1. Add selectable normalization options in config/runtime (`minmax`, `zscore`, `robust`, `normalize`).
2. Add robust-scaler floor (`min_scale`) to prevent low-variance division explosions without channel suppression.
3. Define operating threshold workflow (normal-score percentile, then TPR/FPR reporting on held-out anomaly slices).
4. Document final reproducible Colab run order (minimal cells only) in runbook.
