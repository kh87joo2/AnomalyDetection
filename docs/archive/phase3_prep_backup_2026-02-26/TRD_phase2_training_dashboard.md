# TRD (Technical Requirements / Design Doc) - Phase 2 Training Pipeline Dashboard [2026-02-22]

## Document Scope Tags

- Previous scope backup: `docs/archive/phase1_5_snapshot_2026-02-22/`
- Current scope: dashboard for training pipeline observability

## 0) Decision Lock (Applied)

- Dashboard runs in static web architecture first (HTML/CSS/Vanilla JS).
- Data interface is file-based JSON contract for first release.
- Existing training/validation scripts are reused; no trainer contract changes in this phase.
- `DashboardSample/dashboard` is the baseline UI architecture.

## 1) Target Repository Additions

```text
training_dashboard/
  index.html
  css/
    main.css
    nodes.css
    animations.css
  js/
    app.js
    nodes.js
    connections.js
    drag.js
    panels.js
  data/
    dashboard-state.json          # generated runtime snapshot
    dashboard-layout.json         # static node/layout config

pipelines/
  export_training_dashboard_state.py
```

## 2) Dashboard Data Contracts

### A) Layout Contract (`dashboard-layout.json`)

- Contains fixed graph structure and node metadata:
  - views
  - canvas size
  - groups
  - nodes (`id`, `label`, `type`, `x`, `y`)
  - connections (`from`, `to`)

### B) Runtime State Contract (`dashboard-state.json`)

- Meta:
  - `run_id`, `timestamp`, `repo_root`
- Node statuses:
  - map of `node_id -> {status, message, updated_at}`
- Checklist:
  - array of `{index, title, passed, detail, hint}`
- Metrics:
  - `patchtst` and `swinmae` loss series
  - config summary (`fs`, `lr`, `epochs`, `mask_ratio`, `amp`)
- Artifacts:
  - checkpoint/scaler/log/backup presence and file size

## 3) Frontend Architecture

### A) Rendering Layers

- `group-layer`: static boxes for logical assistant groups
- `connection-layer` (SVG): animated Bezier flow lines + arrow markers
- `node-layer`: draggable node cards with type/status styles

### B) Interaction Model

- Node drag updates connection paths in real time
- Canvas pan and wheel zoom
- Type filter chips (`pi/function/agent/tool`)
- View tabs (`Training Flow`, `Artifact Gate`, optional history view)

### C) Status Styling

- Node type color + status ring overlay
  - `idle`: neutral blue
  - `running`: cyan pulse
  - `done`: green glow
  - `fail`: red glow

## 4) Backend/Script Integration

### A) Export Script (`export_training_dashboard_state.py`)

- Input sources:
  - checkpoints/artifacts/runs/configs
  - output from `pipelines.validate_training_outputs`
- Output:
  - write `training_dashboard/data/dashboard-state.json`

### B) Validation Interface

- Extend validator with machine-readable output option (`--json-out`) or importable function interface.
- Keep current console checklist output unchanged for CLI users.

## 5) Execution Flow

1. Run training scripts.
2. Run output validator.
3. Run dashboard state export script.
4. Open dashboard static page and verify readiness.

## 6) Testing Strategy

- Unit:
  - state export schema validation
  - checklist parsing and status mapping
- Integration:
  - with real artifact paths, state JSON generation success
- UI smoke:
  - graph loads, connections render, panels show checklist and metrics

## 7) Acceptance Criteria

- Dashboard renders with no manual data edits.
- Checklist items and PASS/FAIL state match validator result.
- Loss charts for both streams are visible.
- Artifact readiness is visually obvious (`ready` vs `missing`).
