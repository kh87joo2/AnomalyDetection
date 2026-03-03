# PRD (Product Requirements Doc) - Phase 3A Batch Anomaly Decision Pipeline [2026-02-26]

## Document Scope Tags

- Previous scope backup (Phase 2): `docs/archive/phase3_prep_backup_2026-02-26/`
- Current scope (Phase 3A): test-data import based batch anomaly decision pipeline
- Deferred next scope (Phase 3B): real-time ingestion runtime

## 0) Decision Lock (Applied)

- Do not start from real-time collection in this phase.
- First release runs anomaly decisions from imported test data files.
- Existing training/model artifacts (checkpoint/scaler/config) are reused as-is.
- Preprocess/window assumptions must stay aligned with training path.
- Decision output is fixed to `normal | warn | anomaly`.
- Initial threshold policy uses configurable fixed values for v0; advanced policy tuning is deferred.
- In `dual` mode, if either FDC or vibration input is missing, batch run must not start (fail-fast).
- Execution strategy is Colab-first for validation in this phase; local GPU migration follows after validated completion.
- Real-time adapters/SSE streaming are explicitly deferred to next phase.

## 1) Product Goal

Provide a batch decision pipeline where users import test datasets and immediately obtain per-window anomaly decisions plus run-level summary, so decision logic is validated before real-time deployment.

## 2) In-Scope Features (Phase 3A)

### A) Test Data Import

- Accept local test files for both streams:
  - FDC: csv/parquet
  - vibration: csv/npy
- Support file-path based execution first (CLI), with optional local API trigger.

### B) Offline/Batch Preprocess + Window Builder

- Apply preprocessing compatible with training-time assumptions.
- Build windows using configurable window/stride.
- Guard malformed input (missing fields, NaN/Inf, timestamp issues).

### C) Batch Scoring

- Load existing artifacts and compute scores from imported test files.
- Reuse existing unified inference/scoring code path.

### D) Decision Engine

- Convert scores to `normal|warn|anomaly` using configurable thresholds.
- Include reason and threshold context for each decision.

### E) Result Export and Dashboard Integration

- Export decision events and summary to JSON/CSV.
- Produce dashboard-consumable snapshot output from batch run.
- Dashboard must represent training and test-data decision pipelines independently.
- Dashboard test-data decision view must visualize score trends as graphs with threshold reference lines rendered in the same chart area.

## 3) Out of Scope (Phase 3A)

- Continuous real-time ingestion from external streams.
- SSE/WS push updates and long-running online service guarantees.
- Model retraining and automatic threshold calibration.
- Multi-user auth/RBAC and external alert routing.

## 4) User Scenarios

- User imports test data after training and runs anomaly decision pipeline.
- User checks which windows are normal/warn/anomaly and why.
- User validates threshold behavior with known abnormal samples.
- User reviews both training pipeline status and test-data decision pipeline status in dashboard tabs.

## 5) Success Criteria

- Batch run completes from imported files without crash.
- End-to-end validation run is confirmed in Colab environment.
- Local GPU runtime config is prepared for post-validation migration with the same contracts.
- Decision output includes state (`normal|warn|anomaly`) and reason fields.
- Batch result files are generated and readable for downstream dashboard/report use.
- In `dual` mode, run start is blocked with clear error when one stream input is missing.
- Dashboard presents training flow and test-data decision flow as independent tabs/views.
- Test-data decision dashboard view shows time-series score graphs where score curves and threshold lines are overlaid in the same chart area, so users can visually compare below/above threshold behavior.
- Integration test verifies anomaly-like fixture produces warn/anomaly decisions.

## 6) References

- Existing training/inference code:
  - `trainers/train_patchtst_ssl.py`
  - `trainers/train_swinmae_ssl.py`
  - `inference/scoring.py`
- Existing dashboard/export code:
  - `training_dashboard/`
  - `pipelines/export_training_dashboard_state.py`
