# TRD (Technical Requirements / Design Doc) - Phase 3A Batch Anomaly Decision Pipeline [2026-02-26]

## Document Scope Tags

- Previous scope backup: `docs/archive/phase3_prep_backup_2026-02-26/`
- Current scope: test-data import and batch anomaly decisions
- Deferred: real-time ingestion runtime

## 0) Decision Lock (Applied)

- This phase is batch-run oriented, not continuous stream processing.
- Existing model/checkpoint/scaler contracts stay unchanged.
- Preprocess/normalization/windowing logic must reuse training-compatible paths.
- Runtime entrypoint is CLI-first; API trigger is optional local helper.
- Validation execution is Colab-first, then migrate to local GPU without changing contracts.
- Threshold policy is config-driven and loaded per run.
- v0 threshold policy is fixed-value configurable thresholds with later replacement hooks.
- `dual` mode requires both stream inputs; missing one stream must fail before scoring starts.

## 1) Target Repository Additions

```text
batch_decision/
  __init__.py
  contracts.py                 # batch input/output contracts
  importers.py                 # file loaders for fdc/vibration test files
  preprocess.py                # batch preprocess/window build wrappers
  scoring_engine.py            # artifact load + score orchestration
  decision_engine.py           # threshold policy and state mapping
  runner.py                    # end-to-end batch run entrypoint
  reporting.py                 # json/csv summary export
  service.py                   # optional local API trigger for batch run

dashboard_bridge/
  export_batch_decision_state.py  # generate dashboard state for test-data decision flow

configs/
  batch_decision_runtime_colab.yaml      # Colab validation profile
  batch_decision_runtime_local_gpu.yaml  # local GPU migration profile

artifacts/thresholds/
  batch_decision_thresholds.json  # stream/fused thresholds

tests/batch_decision/
  test_importers.py
  test_decision_engine.py
  test_runner_smoke.py
  test_reporting.py
```

## 2) Runtime Contracts

### A) Batch Input Contract

- Job-level fields:
  - `run_id`
  - `stream`: `patchtst | swinmae | dual`
  - `input_paths`
  - artifact paths (checkpoint/scaler/config)

### B) Score Contract

- Per-window fields:
  - `event_id`
  - `timestamp`
  - `stream_scores`
  - `fused_score`
  - `aux`

### C) Decision Contract

- Per-window fields:
  - `decision`: `normal | warn | anomaly`
  - `reason`
  - `thresholds`
- Run-level summary:
  - counts by decision state
  - max/mean score statistics
- Chart-ready series:
  - ordered score series (stream and fused)
  - threshold reference series/values for `warn` and `anomaly`

## 3) Processing Flow

1. Load test files from CLI/API request.
2. Validate and preprocess stream data.
3. Build windows with configured stride.
4. Score windows via existing inference path.
5. Apply decision policy.
6. Export events + summary artifacts.
7. Emit dashboard snapshot for test-data decision flow.

## 4) Interface Surface (Batch Decision)

- CLI:
  - `python -m batch_decision.runner --config configs/batch_decision_runtime.yaml --input ...`
- Optional local API:
  - `POST /api/batch/run`
  - `GET /api/batch/status/<run_id>`
  - `GET /api/batch/result/<run_id>`

## 5) Config Strategy

- `configs/batch_decision_runtime_colab.yaml` and `configs/batch_decision_runtime_local_gpu.yaml` contain:
  - artifact paths
  - stream mode and input paths
  - window settings
  - threshold policy path
  - output/report paths
- `artifacts/thresholds/batch_decision_thresholds.json` stores initial v0 threshold values and is editable without code changes.


## 6A) Environment Execution Strategy

- Phase 3A validation environment: Google Colab (GPU runtime).
- Migration target after validation: local GPU workstation.
- Keep artifact/input/output schema identical across both profiles to minimize migration diff.

## 6) Dashboard Integration Strategy

- Keep training pipeline and test-data decision pipeline independent.
- Add a separate dashboard tab/view for batch decision flow nodes.
- Keep state files separated to avoid overwrite and preserve history.
- In test-data decision tab/view, render score trend charts with threshold lines overlaid in the same chart so users can visually inspect crossings and margin.

## 7) Testing Strategy

- Unit:
  - import/validation rules
  - decision threshold boundary behavior
- Integration:
  - batch runner from sample files to decision exports
- Smoke:
  - deterministic fixture with expected decision counts

## 8) Acceptance Criteria

- Batch runner executes end-to-end on imported test files.
- Decision outputs and summary exports are generated as configured.
- Tests verify boundary decisions and export schema correctness.
- Dual mode validation blocks run start when either stream input is missing.
- Dashboard displays training flow and batch decision flow in separate tabs/views.
- Dashboard displays score trend graph(s) with threshold reference line(s) overlaid in the same chart for test-data decisions.
- Colab profile end-to-end run is reproducible, and local GPU profile is migration-ready with equivalent contracts.
- Real-time ingestion code is not required for this phase completion.
