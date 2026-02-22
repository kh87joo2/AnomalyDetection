# Todo (Phase 2 Execution Plan - Training Pipeline Dashboard) [2026-02-22]

## Document Scope Tags

- Previous scope backup: `docs/archive/phase1_5_snapshot_2026-02-22/`
- Current execution scope: dashboard implementation for training observability

## 0) Baseline Status

1. Training pipelines (PatchTST/SwinMAE) and output validator are available.
2. Validation checklist script exists: `pipelines/validate_training_outputs.py`.
3. Dashboard sample implementation exists under `DashboardSample/dashboard`.

## 1) Decision Lock for Phase 2

1. Dashboard-first phase only; fusion/threshold implementation is deferred.
2. Reuse sample architecture and evolve with minimal breakage.
3. Start with file-based runtime JSON; backend streaming is deferred.
4. Keep existing training commands and configs unchanged.

## 2) P0 - Dashboard Skeleton

1. Create `training_dashboard/` with `index.html`, `css/`, `js/`, `data/`.
2. Port baseline visual system from `DashboardSample/dashboard`.
3. Define fixed views and node graph layout for training flow.

## 3) P0 - Runtime Data Export

1. Add `pipelines/export_training_dashboard_state.py`.
2. Build `dashboard-state.json` from:
   - checkpoints
   - artifacts
   - runs
   - final configs
   - validation checklist results
3. Add schema checks to prevent malformed state output.

## 4) P0 - Checklist and Metrics Panels

1. Add right panel for checklist (`[v]/[ ]`, PASS/FAIL, detail/hint).
2. Add summary cards:
   - checkpoints ready
   - scaler ready
   - logs ready
   - backup ready
3. Add loss trend chart sections for patchtst/swinmae.

## 5) P1 - Run History and Comparison

1. Persist per-run snapshots (`data/runs/<run_id>.json`).
2. Add run selector UI.
3. Show delta summary between current and selected baseline run.

## 6) P1 - UX Polish

1. Improve status animation by node state.
2. Add quick links to artifact paths/log locations.
3. Add mobile-safe fallback layout for narrow screens.

## 7) P2 - Deferred

1. Real-time stream updates via SSE/WS.
2. Multi-user auth and role controls.
3. Fusion/threshold operational dashboard section.

## 8) Exit Criteria

1. Dashboard loads from generated runtime JSON without manual edits.
2. Checklist parity with validator CLI output is confirmed.
3. Both stream loss trends and artifact readiness are visible.
4. Run history view supports at least one baseline comparison.

## 9) Immediate Next Task (First Build Slice)

1. Scaffold `training_dashboard/` structure.
2. Implement state export script that writes `training_dashboard/data/dashboard-state.json`.
3. Render static training flow with live checklist panel from exported JSON.
