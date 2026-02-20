# PRD (Product Requirements Doc) - Phase 1.5 Real-Data Connection (Reduced Scope) [2026-02-20]

## Document Scope Tags

- Legacy baseline (Phase 1): before 2026-02-20
- Current scope (Phase 1.5): 2026-02-20
- Legacy file: `docs/archive/phase1_legacy/PRD_phase1.md`

## 0) Decision Lock (Applied)

- Keep training entry points unchanged:
  - `python -m trainers.train_patchtst_ssl --config ...`
  - `python -m trainers.train_swinmae_ssl --config ...`
- Add real-data support only inside `build_fdc_datasets()` and `build_vibration_datasets()` via `data.source` branching.
- Reader layer must be non-mutating (as-is load only): no auto sort, no interpolation, no imputation.
- DQVL-lite is required for real-data path and must separate `hard_fail` and `warning`.
- DQ decision is file-level only in Phase 1.5: `keep | drop`.
- Leakage prevention is fixed: split by time first, then windowing; never create cross-file windows.
- FS mismatch policy is fixed:
  - `data.resample.enabled=false` -> mismatch is hard fail (drop)
  - `data.resample.enabled=true` -> only configured deterministic resample is allowed
- Backward compatibility is fixed: keep existing config keys used by current trainers/builders and add only minimal new keys.

## 1) Product Goal

Connect real FDC/vibration data to existing PatchTST-SSL and SwinMAE-SSL pipelines without changing trainer entry points, while preventing low-quality data from contaminating training.

## 2) In-Scope Features (Phase 1.5)

### A) Data Contract

- Add `datasets/contracts.md` with required columns, accepted formats, and minimal examples.
- FDC contract: timestamp + multivariate process parameters.
- Vibration contract: x/y/z + timestamp or sample index + fs metadata rule.

### B) Real-Data Readers

- Add readers:
  - `datasets/readers/fdc_reader.py`
  - `datasets/readers/vib_reader.py`
- Supported input for this phase:
  - FDC: CSV/Parquet
  - Vibration: CSV/NPY

### C) DQVL-lite (Minimum)

- Add DQVL modules:
  - `dqvl/fdc_rules.py`
  - `dqvl/vib_rules.py`
  - `dqvl/report.py`
- Required output: decision-focused JSON with `hard_fails`, `warnings`, and metrics.

### D) Existing Dataset Builder Integration

- `datasets/fdc_dataset.py`: add `source=synthetic|csv|parquet` branch.
- `datasets/vib_dataset.py`: add `source=synthetic|csv|npy` branch.
- Preserve existing train/val flow and normalization policy (`fit on train only`).

### E) Real-Data Smoke Validation

- Add minimal smoke tests with dummy files:
  - `tests/test_fdc_csv_smoke.py`
  - `tests/test_vib_csv_smoke.py`
- Verify one-batch forward + finite loss.

## 3) Out of Scope (Phase 1.5)

- New pipeline entry points (`pipelines/run_*.py`)
- Full DQ score engine (weighted quality score optimization)
- Automatic FS estimation-based resampling
- Large-data optimization (chunking/memmap/distributed)
- Dashboard/report UI

## 4) User Scenarios

- User sets `data.source=csv` and `data.path=...` and runs existing trainer command without code changes.
- System drops clearly invalid files via DQVL hard-fail and records why.
- System warns on recoverable quality issues while continuing if policy allows.
- User can run synthetic and real-data modes from the same codebase.

## 5) Success Criteria

- Existing synthetic path remains operational and unchanged in CLI usage.
- Real-data path runs end-to-end for both streams with same trainer entry points.
- DQVL report JSON is generated with required schema fields.
- No train/val leakage from split/window order violations.
- No windows are generated across file boundaries.
- Smoke tests for CSV/NPY loaders pass with finite training loss.

## 6) References

- PatchTST: https://github.com/yuqinie98/PatchTST
- PatchTST paper: https://arxiv.org/abs/2211.14730
- Swin-MAE paper: https://arxiv.org/abs/2212.13805
- Swin-MAE code: https://github.com/Zian-Xu/Swin-MAE
