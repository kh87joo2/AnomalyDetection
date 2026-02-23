# IMPLEMENTATION_PLAN.md

This plan tracks setup-generated work items and build progress.
The Ralph loop managed plan section is appended or updated automatically.

## Manual Notes

- Add project notes outside the managed block.

<!-- ralph-loop:managed:start -->
## Ralph Loop Managed Plan

### Source Documents (Read-Only)
- `PRD.md`
- `TRD.md`
- `Todo.md`
- `README.md`

### Execution Scope
- In scope:
  - `Todo.md` section 2) P0 - Dashboard Skeleton
  - `Todo.md` section 3) P0 - Runtime Data Export
  - `Todo.md` section 4) P0 - Checklist and Metrics Panels
  - `Todo.md` section 5) P1 - Run History and Comparison
  - `Todo.md` section 6) P1 - UX Polish
- Out of scope:
  - `Todo.md` section 7) P2 - Deferred
  - Fusion/threshold automation and operational alert workflows (`PRD.md` section 3)

### Priority Queue
- [x] P1: Implement P0 Dashboard Skeleton
- [x] P2: Implement P0 Runtime Data Export
- [x] P3: Implement P0 Checklist and Metrics Panels
- [x] P4: Implement P1 Run History and Comparison
- [x] P5: Implement P1 UX Polish

### Completed Baseline
- [x] C1: Backup previous scope snapshot (`docs/archive/phase1_5_snapshot_2026-02-22/`)

### Generated Specs
- `specs/generated/README.md`
- `specs/generated/implement-current-execution-scope-dashboard-implementation-for-training-observability.md`
- `specs/generated/implement-validation-checklist-results.md`
- `specs/generated/implement-previous-scope-backup-docs-archive-phase1-5-snapshot-2026-02-22.md`

### Task Cards
#### T01 - P1 Implement P0 Dashboard Skeleton
- Source requirements:
  - `Todo.md` section 2
  - `PRD.md` section 2.A
  - `TRD.md` sections 1, 3
- Acceptance Criteria:
  - `training_dashboard/index.html` loads `training_dashboard/js/app.js` and renders nodes/edges from `training_dashboard/data/dashboard-layout.json` without editing `index.html` to add nodes.
  - Graph canvas includes `group-layer`, `connection-layer`, and `node-layer`.
  - `training_dashboard/data/dashboard-layout.json` defines nodes and edges for the training flow stages.
- Deliverables:
  - `training_dashboard/index.html`
  - `training_dashboard/css/main.css`
  - `training_dashboard/css/nodes.css`
  - `training_dashboard/css/animations.css`
  - `training_dashboard/js/app.js`
  - `training_dashboard/js/nodes.js`
  - `training_dashboard/js/connections.js`
  - `training_dashboard/js/drag.js`
  - `training_dashboard/data/dashboard-layout.json`
- Verify:
  - `test -f training_dashboard/index.html`
  - `test -f training_dashboard/data/dashboard-layout.json`
  - `grep -q "group-layer" training_dashboard/index.html && grep -q "connection-layer" training_dashboard/index.html && grep -q "node-layer" training_dashboard/index.html`
  - `python -m http.server 8000`

#### T02 - P2 Implement P0 Runtime Data Export
- Source requirements:
  - `Todo.md` section 3
  - `TRD.md` sections 2.B, 4
- Acceptance Criteria:
  - `pipelines/export_training_dashboard_state.py` generates `training_dashboard/data/dashboard-state.json`.
  - Output contains required top-level keys: `meta`, `nodes`, `checklist`, `metrics`, `artifacts`.
  - Export logic reuses existing validator module `pipelines/validate_training_outputs.py` (import/function call), not ad-hoc duplicated checks.
  - Export fails with actionable error when required inputs are missing or malformed.
- Deliverables:
  - `pipelines/__init__.py` (package entry maintained for `python -m pipelines.*`)
  - `pipelines/export_training_dashboard_state.py`
  - `training_dashboard/data/dashboard-state.json` (generated sample)
  - `tests/pipelines/test_export_training_dashboard_state.py`
- Verify:
  - `test -f pipelines/__init__.py`
  - `python -m pipelines.export_training_dashboard_state --repo-root . --out training_dashboard/data/dashboard-state.json`
  - `python -m json.tool training_dashboard/data/dashboard-state.json >/dev/null`
  - `pytest -q tests/pipelines/test_export_training_dashboard_state.py`

#### T03 - P3 Implement P0 Checklist and Metrics Panels
- Source requirements:
  - `Todo.md` section 4
  - `PRD.md` sections 2.B, 2.C
  - `TRD.md` section 7
- Acceptance Criteria:
  - Dashboard right panel renders checklist rows for required item set (`checkpoints`, `scaler`, `logs`, `configs`, `backup`, `smoke`, `split_policy`) with `[v]/[ ]`, `PASS/FAIL`, detail, hint.
  - Summary cards render `checkpoints`, `scaler`, `logs`, `backup` ready/missing state from exported data.
  - PatchTST and SwinMAE train/val loss trend visualizations render from `dashboard-state.json`.
- Deliverables:
  - `pipelines/validate_training_outputs.py` (reuse only, no new validator file)
  - `training_dashboard/js/panels.js`
  - `training_dashboard/js/app.js` (state binding)
  - `training_dashboard/css/main.css` (panel layout updates)
  - `training_dashboard/css/nodes.css` (status color sync)
- Verify:
  - `python -m pipelines.validate_training_outputs --repo-root . --skip-smoke`
  - `python -m pipelines.export_training_dashboard_state --repo-root . --out training_dashboard/data/dashboard-state.json`
  - `python -m http.server 8000`

#### T04 - P4 Implement P1 Run History and Comparison
- Source requirements:
  - `Todo.md` section 5
  - `PRD.md` section 2.D
- Acceptance Criteria:
  - Per-run snapshots are stored under `training_dashboard/data/runs/<run_id>.json`.
  - `run_id` is normalized to `[a-zA-Z0-9._-]` (`/`, whitespace, and other symbols are replaced with `_`).
  - Dashboard can select current vs baseline run from local snapshots.
  - Comparison panel shows at least checklist pass count delta and final loss delta per stream.
- Deliverables:
  - `pipelines/export_training_dashboard_state.py` (run snapshot mode)
  - `training_dashboard/js/panels.js` (run selector and delta view)
  - `training_dashboard/data/runs/` (sample snapshots)
  - `tests/pipelines/test_dashboard_run_history.py`
- Verify:
  - `python -m pipelines.export_training_dashboard_state --repo-root . --run-id demo --persist-run-history`
  - `test -d training_dashboard/data/runs`
  - `pytest -q tests/pipelines/test_dashboard_run_history.py`

#### T05 - P5 Implement P1 UX Polish
- Source requirements:
  - `Todo.md` section 6
  - `TRD.md` section 3.C
- Acceptance Criteria:
  - Node status animation reflects `idle/running/done/fail` consistently.
  - Quick links to relevant artifact/log paths are visible and clickable.
  - Layout remains usable on narrow screens (mobile-safe fallback).
- Deliverables:
  - `training_dashboard/css/animations.css`
  - `training_dashboard/css/main.css`
  - `training_dashboard/js/panels.js`
  - `training_dashboard/js/connections.js`
- Verify:
  - `python -m http.server 8000`
  - Manual smoke check at widths `>=1280px` and `<=768px`
  - `ruff check training_dashboard pipelines tests`

### Open Questions
- [ ] Q01: Dashboard serving strategy for local operation - `python -m http.server` only vs lightweight `FastAPI` static mount? (decision needed before T05; default=`http.server`; impacts packaging and runbook)
- [ ] Q02: Run history storage policy - keep all snapshots vs retain latest N with pruning? (decision needed before T04; default=`keep latest 20`; impacts disk usage and reproducibility)
- [ ] Q03: Chart rendering strategy - pure SVG/canvas custom plots vs lightweight chart dependency? (decision needed before T03; default=`dependency-free canvas line plot`; impacts maintainability and bundle size)

### Loop Rules
- Setup mode changes planning artifacts (`AGENTS.md`, `IMPLEMENTATION_PLAN.md`, `specs/generated/*.md`, `PROMPT_plan.md`, `PROMPT_build.md`, `loop.sh`, `.ralph-loop/*.py`).
- Build mode executes exactly one `[ ]` task per cycle.
- Keep source docs (`PRD.md`, `TRD.md`, `Todo.md`, `docs/*`, `README.md`) read-only during setup.
<!-- ralph-loop:managed:end -->
