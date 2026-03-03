# Todo (Phase 3A Execution Plan - Batch Anomaly Decision Pipeline) [2026-02-26]

## Document Scope Tags

- Previous scope backup: `docs/archive/phase3_prep_backup_2026-02-26/`
- Current execution scope: import test data and run batch anomaly decisions
- Deferred next scope: real-time ingestion runtime

## 0) Baseline Status

1. Training and unified inference/scoring code already exist.
2. Dashboard output/export path already exists.
3. Real-time runtime is not required for this phase.

## 1) Decision Lock for Phase 3A

1. Start from test data import, not live collection.
2. Reuse existing artifacts and model contracts.
3. Build CLI-first batch decision pipeline, API optional.
4. Reuse training-compatible preprocess/window contracts.
5. Keep decision policy threshold-configurable (v0 uses fixed configurable values).
6. In `dual` mode, if either stream input is missing, batch run must not start.
7. Validate in Colab first, then prepare local GPU migration profile.

## 2) P0 - Batch Decision Skeleton

1. Add `batch_decision/` package skeleton and contracts.
2. Add `configs/batch_decision_runtime.yaml` and threshold template.
3. Add minimal runner entrypoint with config load validation.

## 2A) P0 - Colab Validation Profile

1. Add Colab runtime config profile for batch decision execution.
2. Add reproducible Colab run steps in docs/notebook notes.
3. Validate one end-to-end Colab execution path.

## 3) P0 - Test Data Import and Preprocess

1. Implement importers for FDC/vibration test files.
2. Implement batch preprocessing + window builder wrappers by reusing training-time transforms/contracts.
3. Add validation handling for malformed input.

## 4) P0 - Batch Scoring Integration

1. Load checkpoint/scaler/config artifacts for batch runs.
2. Reuse `inference/scoring.py` for per-window scoring.
3. Emit unified per-window score payload.
4. Add fail-fast validation for `dual` mode missing-stream input.

## 5) P1 - Decision and Reporting

1. Implement threshold decision engine (`normal|warn|anomaly`).
2. Add reason generation and summary statistics.
3. Export decision events and summary to JSON/CSV.
4. Seed initial v0 thresholds in config/artifact file with easy update path.
5. Export chart-ready score trend payload (ordered scores + threshold references).

## 6) P1 - API Trigger and Dashboard Bridge

1. Add optional local API trigger for batch run execution.
2. Add result/status retrieval endpoints.
3. Add test-data decision dashboard state export and separate dashboard tab/view node flow.
4. Add score trend graph in the test-data decision tab/view with score and threshold lines overlaid in the same chart.

## 6A) P1 - Local GPU Migration Readiness

1. Add local GPU runtime profile with same schema/contracts as Colab.
2. Verify path/device toggles are config-only changes.
3. Document Colab-to-local migration checklist.

## 7) P2 - Deferred (Next Phase)

1. Continuous real-time ingestion adapters.
2. SSE/WS live stream updates.
3. External alerting and production hardening.

## 8) Exit Criteria

1. Imported test files can be processed end-to-end.
2. Decisions are generated with clear reason fields.
3. Exported artifacts are usable for dashboard/report flow.
4. Dashboard has independent tabs/views for training flow and test-data decision flow.
5. Dashboard graph clearly shows score trend vs threshold lines in the same chart (below/above threshold visible).
6. Colab end-to-end run is verified and reproducible.
7. Local GPU migration profile is prepared with equivalent contracts.
8. Integration test validates known abnormal fixture behavior.

## 9) Immediate Next Task (First Build Slice)

1. Scaffold `batch_decision/` package and runtime config files.
2. Implement runner config validation and dry-run mode.
3. Add first tests for config/import contract checks.
