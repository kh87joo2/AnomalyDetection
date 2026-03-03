# PRD (Product Requirements Doc) - Phase 2 Training Pipeline Dashboard [2026-02-22]

## Document Scope Tags

- Previous scope (Phase 1.5): real-data connection and Colab execution
- Current scope (Phase 2): training pipeline dashboard
- Backup snapshot: `docs/archive/phase1_5_snapshot_2026-02-22/`

## 0) Decision Lock (Applied)

- This phase focuses only on training pipeline visualization and run validation UX.
- Fusion/threshold automation is explicitly out of scope for this phase.
- Dashboard foundation reuses `DashboardSample/dashboard` interaction pattern (HTML/CSS/JS + SVG).
- First release can run in static/local mode (file-based data input), without backend service.
- Existing training CLI and config contracts remain unchanged.

## 1) Product Goal

Provide a single dashboard where users can verify training execution status, key loss trends, and output readiness gates without manually checking multiple folders/logs.

## 2) In-Scope Features (Phase 2)

### A) Training Graph View

- Node graph view for training pipeline stages:
  - data prep
  - dqvl
  - patchtst training
  - swinmae training
  - artifact save
  - validation gate
- Node status rendering: `idle | running | done | fail`.
- Connection flow animation and node-type legend/filter.

### B) Checklist Gate View

- Render the 7-item validation checklist from `pipelines.validate_training_outputs`.
- Show each item with `[v]/[ ]`, `PASS/FAIL`, detail message, and remediation hint.
- Show summary count: `passed / total`.

### C) Loss & Run Summary Panel

- Show train/val loss trend snapshots for PatchTST and SwinMAE.
- Show final config highlights (`fs`, `lr`, `epochs`, `mask_ratio`, `amp`).
- Show artifact existence quick status (checkpoint/scaler/runs/backup).

### D) Run History (Basic)

- Support selecting a run record from local run snapshots.
- Compare high-level metrics between selected runs.

## 3) Out of Scope (Phase 2)

- Real-time distributed orchestration backend (SSE/WS mandatory integration).
- Multi-user auth/RBAC, external database, tenant isolation.
- Fusion score generation and threshold calibration workflow implementation.
- Alert routing, incident workflow, and report publishing automation.

## 4) User Scenarios

- User finishes training and opens dashboard to confirm outputs are ready.
- User quickly identifies which validation checklist item failed and why.
- User compares current run against previous run before promoting artifacts.

## 5) Success Criteria

- Dashboard loads and renders the training node graph and checklist without manual HTML edits.
- Validation results can be refreshed from generated run state data.
- Both model streams' loss trend is visible on one screen.
- User can determine release readiness within 1 minute of opening dashboard.

## 6) References

- Dashboard sample: `DashboardSample/dashboard/`
- Training output validator: `pipelines/validate_training_outputs.py`
- Existing training pipelines:
  - `trainers/train_patchtst_ssl.py`
  - `trainers/train_swinmae_ssl.py`
