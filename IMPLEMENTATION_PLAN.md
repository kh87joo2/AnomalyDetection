# IMPLEMENTATION_PLAN.md

This plan tracks setup-generated work items and build progress.
The Ralph loop managed plan section is appended or updated automatically.

## Manual Notes

- 2026-02-26 pre-start sync:
  - Added Colab-first validation stage and local GPU migration-readiness stage to the active Phase 3A plan.
  - Reflected user constraints: no real-time-first, dual-mode missing-stream fail-fast, threshold config v0 fixed/editable.
  - Locked dashboard direction: training flow and test-data decision flow are separate tabs/views, and test graph overlays score+threshold lines in one chart area.
  - Kept implementation paused until explicit user start signal.

<!-- ralph-loop:managed:start -->
## Ralph Loop Managed Plan

### Source Documents (Read-Only)
- `prd.md`
- `trd.md`
- `todo.md`
- `README.md`

### Priority Queue
- [x] P0A: Back up previous Phase 2 docs to `docs/archive/phase3_prep_backup_2026-02-26/`.
- [x] P0B: Build batch decision skeleton (`batch_decision/` package + dry-run runner).
- [x] P0C: Add Colab validation profile and reproducible Colab execution path.
- [x] P0D: Implement test-data import + training-compatible preprocess/window builder.
- [x] P0E: Integrate batch scoring with existing checkpoints/scaler/config.
- [x] P1A: Implement decision engine + reporting exports (JSON/CSV + reason fields).
- [x] P1B: Implement dashboard bridge with separate test-data decision tab/view and overlaid score/threshold graph.
- [ ] P1C: Prepare local GPU migration profile (config-only toggles from Colab profile).

### Generated Specs
- `specs/generated/README.md`
- `specs/generated/3-p0-test-data-import-and-preprocess.md`
- `specs/generated/current-execution-scope-import-test-data-and-run-batch-anomaly-decisions.md`
- `specs/generated/implement-2-p0-batch-decision-skeleton.md`
- `specs/generated/implement-2a-p0-colab-validation-profile.md`
- `specs/generated/implement-4-p0-batch-scoring-integration.md`
- `specs/generated/implement-5-p1-decision-and-reporting.md`
- `specs/generated/implement-deferred-next-scope-real-time-ingestion-runtime.md`
- `specs/generated/implement-previous-scope-backup-docs-archive-phase3-prep-backup-2026-02-26.md`

### Task Cards
#### T01 - P0B Batch Decision Skeleton
- Source: `todo.md`
- Done checklist:
  - [x] Behavior is implemented and mapped to source requirements.
  - [x] Acceptance criteria are explicit and testable.
  - [x] Tests are added or updated for this task.
  - [x] Verification commands in `AGENTS.md` pass.

#### T02 - P0C Colab Validation Profile
- Source: `todo.md`
- Done checklist:
  - [x] Behavior is implemented and mapped to source requirements.
  - [x] Acceptance criteria are explicit and testable.
  - [x] Tests are added or updated for this task.
  - [x] Verification commands in `AGENTS.md` pass.

#### T03 - P0D Import + Preprocess
- Source: `todo.md`
- Done checklist:
  - [x] Behavior is implemented and mapped to source requirements.
  - [x] Acceptance criteria are explicit and testable.
  - [x] Tests are added or updated for this task.
  - [x] Verification commands in `AGENTS.md` pass.

#### T04 - P0E Batch Scoring Integration
- Source: `todo.md`
- Done checklist:
  - [x] Behavior is implemented and mapped to source requirements.
  - [x] Acceptance criteria are explicit and testable.
  - [x] Tests are added or updated for this task.
  - [x] Verification commands in `AGENTS.md` pass.

#### T05 - P1A Decision + Reporting
- Source: `todo.md`
- Done checklist:
  - [x] Behavior is implemented and mapped to source requirements.
  - [x] Acceptance criteria are explicit and testable.
  - [x] Tests are added or updated for this task.
  - [x] Verification commands in `AGENTS.md` pass.

#### T06 - P1B Dashboard Bridge
- Source: `todo.md`
- Done checklist:
  - [x] Behavior is implemented and mapped to source requirements.
  - [x] Acceptance criteria are explicit and testable.
  - [x] Tests are added or updated for this task.
  - [x] Verification commands in `AGENTS.md` pass.

#### T07 - P1C Local GPU Migration Readiness
- Source: `todo.md`
- Done checklist:
  - [ ] Behavior is implemented and mapped to source requirements.
  - [ ] Acceptance criteria are explicit and testable.
  - [ ] Tests are added or updated for this task.
  - [ ] Verification commands in `AGENTS.md` pass.

### Open Questions
- [x] Q01: Dashboard tab naming for test-data decision flow resolved as `Batch Decision`.

### Loop Rules
- Setup mode changes planning artifacts (`AGENTS.md`, `IMPLEMENTATION_PLAN.md`, `specs/generated/*.md`, `PROMPT_plan.md`, `PROMPT_build.md`, `loop.sh`, `.ralph-loop/*.py`).
- Build mode executes exactly one `[ ]` task per cycle.
- Keep source docs (`prd.md`, `trd.md`, `todo.md`, `docs/*`, `README.md`) read-only during setup.
<!-- ralph-loop:managed:end -->
