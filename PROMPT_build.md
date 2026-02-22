# PROMPT_build.md

Mode: BUILDING
Goal: Complete exactly one highest-priority unchecked task.

Rules:
1. Pick one `[ ]` task from `IMPLEMENTATION_PLAN.md` and mark it `[-]`.
2. Implement only the selected task scope.
3. Update tests for the changed behavior.
4. Run verification commands listed in `AGENTS.md`.
5. Mark task `[x]` only after verification passes.
6. If blocked, keep `[-]` and document blockers in `Open Questions`.

Output:
- Code changes for one task
- Updated task state in `IMPLEMENTATION_PLAN.md`
- Verification summary
