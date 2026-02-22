# PROMPT_plan.md

Mode: PLANNING
Goal: Refresh planning artifacts from requirements without editing source code.

Rules:
1. Treat `prd.md`, `trd.md`, `todo.md`, `docs/*`, and human-authored `specs/*.md` as read-only.
2. Update only `AGENTS.md`, `IMPLEMENTATION_PLAN.md`, and generated planning artifacts.
3. Keep tasks small, testable, and independently completable.
4. Keep one source-traceable task card per priority item.
5. Surface unresolved ambiguity in `Open Questions` with impact tags.

Output:
- Updated `IMPLEMENTATION_PLAN.md`
- Updated `AGENTS.md`
- Updated `specs/generated/*.md` as needed
