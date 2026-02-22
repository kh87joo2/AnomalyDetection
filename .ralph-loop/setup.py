#!/usr/bin/env python3
from __future__ import annotations

import argparse
import dataclasses
import difflib
import json
import pathlib
import re
import sys
import textwrap
from typing import Iterable


MANAGED_START = "<!-- ralph-loop:managed:start -->"
MANAGED_END = "<!-- ralph-loop:managed:end -->"

ROOT_PRIORITY = [
    ("prd.md", 120),
    ("trd.md", 115),
    ("todo.md", 130),
]

DOCS_PRIORITY = [
    ("docs/prd.md", 100),
    ("docs/trd.md", 95),
    ("docs/todo.md", 110),
]

HEADING_RE = re.compile(r"^\s{0,3}#{2,6}\s+(.+?)\s*$")
OPEN_TASK_RE = re.compile(r"^\s*[-*]\s+\[\s\]\s+(.+?)\s*$")
BULLET_RE = re.compile(r"^\s*[-*]\s+(.+?)\s*$")
LINK_RE = re.compile(r"\[(.*?)\]\(.*?\)")
CODE_RE = re.compile(r"`([^`]*)`")
SPACE_RE = re.compile(r"\s+")
VERB_RE = re.compile(
    r"\b(add|build|create|define|design|document|implement|improve|integrate|"
    r"migrate|refactor|remove|support|test|update|write|configure|setup|set up)\b",
    re.IGNORECASE,
)
QUESTION_RE = re.compile(
    r"\?|(?:\b(?:tbd|todo|unknown|decide|clarify|pending|to be decided)\b)",
    re.IGNORECASE,
)
ACCEPTANCE_RE = re.compile(
    r"(acceptance criteria|done when|definition of done|완료 조건)",
    re.IGNORECASE,
)
KEYWORD_BONUS_RE = re.compile(
    r"\b(must|required|critical|blocker|p0|p1|urgent)\b", re.IGNORECASE
)
SKIP_HEADINGS = {
    "overview",
    "introduction",
    "background",
    "notes",
    "appendix",
    "references",
    "changelog",
    "summary",
}

IMPACT_KEYWORDS = [
    ("API", ("api", "endpoint", "http", "grpc", "graphql", "contract", "interface")),
    ("Data schema", ("schema", "database", "db", "migration", "table", "model", "data")),
    ("Test criteria", ("test", "qa", "acceptance", "unit", "integration", "e2e")),
    ("Security", ("auth", "oauth", "permission", "security", "token", "secret")),
]


@dataclasses.dataclass(frozen=True)
class SourceDoc:
    path: pathlib.Path
    relpath: str
    priority: int
    text: str


@dataclasses.dataclass(frozen=True)
class Task:
    title: str
    source: str
    priority: int


@dataclasses.dataclass(frozen=True)
class Question:
    text: str
    source: str
    impacts: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class SpecEntry:
    slug: str
    task: Task


@dataclasses.dataclass(frozen=True)
class FileChange:
    path: str
    state: str
    added: int
    removed: int


def read_text(path: pathlib.Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def find_git_root(start: pathlib.Path) -> pathlib.Path | None:
    for candidate in (start, *start.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


def discover_documents(repo_root: pathlib.Path) -> list[SourceDoc]:
    found: list[SourceDoc] = []
    seen: set[pathlib.Path] = set()

    def add_if_exists(rel: str, priority: int) -> None:
        path = repo_root / rel
        if not path.is_file():
            return
        resolved = path.resolve()
        if resolved in seen:
            return
        seen.add(resolved)
        found.append(
            SourceDoc(
                path=path,
                relpath=path.relative_to(repo_root).as_posix(),
                priority=priority,
                text=read_text(path),
            )
        )

    for rel, priority in ROOT_PRIORITY:
        add_if_exists(rel, priority)
    for rel, priority in DOCS_PRIORITY:
        add_if_exists(rel, priority)

    specs_dir = repo_root / "specs"
    if specs_dir.is_dir():
        for path in sorted(specs_dir.rglob("*.md")):
            rel = path.relative_to(repo_root).as_posix()
            if rel.startswith("specs/generated/"):
                continue
            if rel == "specs/README.md":
                continue
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            found.append(
                SourceDoc(
                    path=path,
                    relpath=rel,
                    priority=90,
                    text=read_text(path),
                )
            )

    add_if_exists("README.md", 50)
    return found


def normalize_text(value: str) -> str:
    value = LINK_RE.sub(r"\1", value)
    value = CODE_RE.sub(r"\1", value)
    value = value.strip(" -*#:_|")
    value = SPACE_RE.sub(" ", value).strip()
    return value


def maybe_actionable(text: str) -> str | None:
    text = normalize_text(text)
    if len(text) < 8:
        return None
    if text.lower() in SKIP_HEADINGS:
        return None
    if not VERB_RE.search(text):
        text = f"Implement {text}"
    return text


def extract_tasks(documents: list[SourceDoc], limit: int = 12) -> list[Task]:
    ranked: list[tuple[int, int, str, str]] = []
    order = 0
    for doc in documents:
        is_todo = doc.path.name.lower() == "todo.md"
        for line in doc.text.splitlines():
            raw = None
            score = doc.priority
            open_task_match = OPEN_TASK_RE.match(line)
            if open_task_match:
                raw = open_task_match.group(1)
                score += 30
            else:
                bullet_match = BULLET_RE.match(line)
                if bullet_match and is_todo:
                    raw = bullet_match.group(1)
                    score += 20
                else:
                    heading_match = HEADING_RE.match(line)
                    if heading_match:
                        raw = heading_match.group(1)

            if raw is None:
                continue

            cleaned = maybe_actionable(raw)
            if cleaned is None:
                continue
            if KEYWORD_BONUS_RE.search(cleaned):
                score += 10
            ranked.append((score, order, cleaned, doc.relpath))
            order += 1

    dedup: dict[str, tuple[int, int, str, str]] = {}
    for score, idx, text, source in ranked:
        key = text.casefold()
        prev = dedup.get(key)
        if prev is None or score > prev[0]:
            dedup[key] = (score, idx, text, source)

    sorted_items = sorted(dedup.values(), key=lambda item: (-item[0], item[1]))[:limit]
    tasks = [Task(title=text, source=source, priority=score) for score, _, text, source in sorted_items]
    if tasks:
        return tasks

    return [
        Task("Define the first vertical slice from project requirements", "generated", 0),
        Task("Implement the vertical slice with tests", "generated", 0),
        Task("Run verification commands and prepare release notes", "generated", 0),
    ]


def slugify(value: str) -> str:
    lowered = value.casefold()
    slug = re.sub(r"[^a-z0-9]+", "-", lowered).strip("-")
    return slug or "task"


def compute_spec_entries(tasks: list[Task], limit: int = 8) -> list[SpecEntry]:
    entries: list[SpecEntry] = []
    used: dict[str, int] = {}
    for task in tasks[:limit]:
        base = slugify(task.title)
        count = used.get(base, 0) + 1
        used[base] = count
        slug = base if count == 1 else f"{base}-{count}"
        entries.append(SpecEntry(slug=slug, task=task))
    return entries


def classify_impacts(text: str) -> tuple[str, ...]:
    lowered = text.casefold()
    impacts: list[str] = []
    for label, keywords in IMPACT_KEYWORDS:
        if any(keyword in lowered for keyword in keywords):
            impacts.append(label)
    if not impacts:
        impacts.append("Architecture")
    return tuple(impacts)


def extract_questions(documents: list[SourceDoc], limit: int = 8) -> list[Question]:
    extracted: list[Question] = []
    seen: set[str] = set()
    for doc in documents:
        for line in doc.text.splitlines():
            if not QUESTION_RE.search(line):
                continue
            cleaned = normalize_text(line)
            if len(cleaned) < 10:
                continue
            key = cleaned.casefold()
            if key in seen:
                continue
            seen.add(key)
            extracted.append(
                Question(text=cleaned, source=doc.relpath, impacts=classify_impacts(cleaned))
            )
            if len(extracted) >= limit:
                return extracted
    return extracted


def missing_document_questions(documents: list[SourceDoc]) -> list[Question]:
    relpaths = {doc.relpath for doc in documents}
    questions: list[Question] = []

    if "prd.md" not in relpaths and "docs/prd.md" not in relpaths:
        questions.append(
            Question(
                "What user outcomes define success for this delivery cycle?",
                "generated",
                ("Test criteria", "Architecture"),
            )
        )
    if "trd.md" not in relpaths and "docs/trd.md" not in relpaths:
        questions.append(
            Question(
                "What technical constraints, interfaces, and dependencies are mandatory?",
                "generated",
                ("API", "Data schema"),
            )
        )
    if "todo.md" not in relpaths and "docs/todo.md" not in relpaths:
        questions.append(
            Question(
                "Which work item should be the first end-to-end slice?",
                "generated",
                ("Architecture", "Test criteria"),
            )
        )

    has_acceptance = any(ACCEPTANCE_RE.search(doc.text) for doc in documents)
    if not has_acceptance:
        questions.append(
            Question(
                "What explicit acceptance criteria should define done for each priority task?",
                "generated",
                ("Test criteria",),
            )
        )
    return questions


def detect_make_targets(makefile: pathlib.Path) -> set[str]:
    if not makefile.is_file():
        return set()
    targets: set[str] = set()
    for line in read_text(makefile).splitlines():
        if line.startswith("\t") or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        name = line.split(":", 1)[0].strip()
        if not name or " " in name or "=" in name:
            continue
        targets.add(name)
    return targets


def detect_verification_commands(repo_root: pathlib.Path) -> list[str]:
    make_targets = detect_make_targets(repo_root / "Makefile")
    commands: list[str] = []
    for target in ("format", "lint", "test"):
        if target in make_targets:
            commands.append(f"make {target}")
    if commands:
        return commands

    package_json = repo_root / "package.json"
    if package_json.is_file():
        try:
            parsed = json.loads(read_text(package_json))
            scripts = parsed.get("scripts", {})
            for script_name in ("format", "lint", "test"):
                if script_name in scripts:
                    commands.append(f"npm run {script_name}")
        except json.JSONDecodeError:
            pass
    if commands:
        return commands

    if (repo_root / "pyproject.toml").is_file() or (repo_root / "tests").is_dir():
        return ["pytest"]

    return [
        "TODO: set project-specific format command",
        "TODO: set project-specific lint command",
        "TODO: set project-specific test command",
    ]


def relevant_questions_for_task(task: Task, questions: list[Question], limit: int = 3) -> list[Question]:
    task_words = {word for word in re.split(r"[^a-z0-9]+", task.title.casefold()) if len(word) > 2}
    if not task_words:
        return questions[:limit]

    ranked: list[tuple[int, Question]] = []
    for question in questions:
        q_words = {
            word
            for word in re.split(r"[^a-z0-9]+", question.text.casefold())
            if len(word) > 2
        }
        overlap = len(task_words.intersection(q_words))
        source_bonus = 2 if question.source == task.source else 0
        score = overlap + source_bonus
        ranked.append((score, question))

    ranked.sort(key=lambda item: item[0], reverse=True)
    selected = [question for score, question in ranked if score > 0][:limit]
    if selected:
        return selected
    return questions[:limit]


def render_generated_spec(entry: SpecEntry, questions: list[Question]) -> str:
    task = entry.task
    lines = [f"# {task.title}", ""]
    lines.append("## Metadata")
    lines.append(f"- slug: `{entry.slug}`")
    lines.append(f"- source: `{task.source}`")
    lines.append("- status: draft")
    lines.append("")
    lines.append("## JTBD")
    lines.append(
        f"- Job: Deliver `{task.title}` so users get the expected outcome described in the source documents."
    )
    lines.append("")
    lines.append("## Scope")
    lines.append(f"- In scope: `{task.title}`")
    lines.append("- Out of scope: unrelated refactors or broad architectural changes.")
    lines.append("")
    lines.append("## Acceptance Criteria")
    lines.append("- [ ] Behavior is implemented and traceable to source requirements.")
    lines.append("- [ ] Tests validate the behavior (unit/integration/e2e as applicable).")
    lines.append("- [ ] Verification commands in `AGENTS.md` pass.")
    lines.append("")
    lines.append("## Dependencies")
    lines.append("- [ ] Identify upstream/downstream interfaces affected by this task.")
    lines.append("- [ ] Confirm required schema/API contracts before implementation.")
    lines.append("")
    lines.append("## Open Questions")
    task_questions = relevant_questions_for_task(task, questions)
    if task_questions:
        for question in task_questions:
            impacts = ", ".join(question.impacts)
            lines.append(f"- [ ] {question.text} (impacts: {impacts}; source: `{question.source}`)")
    else:
        lines.append("- [ ] None")
    lines.append("")
    lines.append("## Notes")
    lines.append("- Generated by `scripts/setup.py` for Ralph loop planning.")
    return "\n".join(lines) + "\n"


def render_generated_specs_index(entries: list[SpecEntry], documents: list[SourceDoc]) -> str:
    lines: list[str] = ["# Generated Specs", ""]
    lines.append("These files are generated from PRD/TRD/TODO inputs for Ralph loop execution.")
    lines.append("")
    lines.append("## Source Documents")
    if documents:
        for doc in documents:
            lines.append(f"- `{doc.relpath}`")
    else:
        lines.append("- none")
    lines.append("")
    lines.append("## Specs")
    for entry in entries:
        lines.append(f"- [`{entry.slug}.md`](./{entry.slug}.md): {entry.task.title}")
    lines.append("")
    lines.append("Regenerate with: `scripts/setup.py .`")
    return "\n".join(lines) + "\n"


def render_prompt_plan() -> str:
    return textwrap.dedent(
        """\
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
        """
    )


def render_prompt_build() -> str:
    return textwrap.dedent(
        """\
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
        """
    )


def render_loop_script() -> str:
    return textwrap.dedent(
        """\
        #!/usr/bin/env bash
        set -euo pipefail

        MODE="${1:-auto}"
        CYCLES="${2:-1}"
        REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
        SETUP_SCRIPT="$REPO_ROOT/.ralph-loop/setup.py"
        NEXT_TASK_SCRIPT="$REPO_ROOT/.ralph-loop/next-task.py"
        AUTO_WIP_COMMIT="${AUTO_WIP_COMMIT:-1}"
        WIP_COMMIT_PREFIX="${WIP_COMMIT_PREFIX:-chore(wip): ralph checkpoint}"

        if [[ ! -f "$SETUP_SCRIPT" ]]; then
          echo "missing $SETUP_SCRIPT. run setup.py once in this repo to bootstrap loop assets."
          exit 2
        fi

        run_setup() {
          python3 "$SETUP_SCRIPT" "$REPO_ROOT"
        }

        run_plan_hook() {
          if [[ -n "${RALPH_PLAN_CMD:-}" ]]; then
            eval "$RALPH_PLAN_CMD \\"$REPO_ROOT/PROMPT_plan.md\\""
          else
            echo "plan hook not set. Set RALPH_PLAN_CMD to invoke your coding agent with PROMPT_plan.md."
          fi
        }

        run_build_hook() {
          if [[ -n "${RALPH_BUILD_CMD:-}" ]]; then
            eval "$RALPH_BUILD_CMD \\"$REPO_ROOT/PROMPT_build.md\\""
          else
            echo "build hook not set. Set RALPH_BUILD_CMD to invoke your coding agent with PROMPT_build.md."
          fi
        }

        print_status() {
          python3 "$NEXT_TASK_SCRIPT" "$REPO_ROOT/IMPLEMENTATION_PLAN.md" || true
        }

        maybe_wip_commit() {
          if [[ "$AUTO_WIP_COMMIT" != "1" ]]; then
            return 0
          fi

          if ! command -v git >/dev/null 2>&1; then
            echo "git not found; skip checkpoint commit."
            return 0
          fi

          if ! git -C "$REPO_ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
            echo "not a git repository; skip checkpoint commit."
            return 0
          fi

          local untracked_files
          untracked_files="$(git -C "$REPO_ROOT" ls-files --others --exclude-standard)"
          if git -C "$REPO_ROOT" diff --quiet && git -C "$REPO_ROOT" diff --cached --quiet && [[ -z "$untracked_files" ]]; then
            echo "no local changes; skip checkpoint commit."
            return 0
          fi

          git -C "$REPO_ROOT" add -A
          if git -C "$REPO_ROOT" diff --cached --quiet; then
            echo "no staged changes after add; skip checkpoint commit."
            return 0
          fi

          local next_task message
          next_task="$(python3 "$NEXT_TASK_SCRIPT" "$REPO_ROOT/IMPLEMENTATION_PLAN.md" 2>/dev/null || true)"
          message="$WIP_COMMIT_PREFIX"
          if [[ -n "$next_task" && "$next_task" != "no open task found" ]]; then
            message="$message | next: $next_task"
          fi

          if git -C "$REPO_ROOT" commit -m "$message"; then
            echo "created checkpoint commit: $message"
          else
            echo "checkpoint commit failed; review git config or set AUTO_WIP_COMMIT=0."
          fi
        }

        case "$MODE" in
          setup|plan)
            run_setup
            run_plan_hook
            ;;
          build)
            echo "next task:"
            print_status
            run_build_hook
            maybe_wip_commit
            ;;
          status)
            echo "next task:"
            print_status
            ;;
          auto)
            for ((i = 1; i <= CYCLES; i++)); do
              echo "cycle $i/$CYCLES"
              run_setup
              run_plan_hook
              echo "next task:"
              print_status
              run_build_hook
              maybe_wip_commit
            done
            ;;
          *)
            echo "usage: $0 [setup|plan|build|auto|status] [cycles]"
            exit 2
            ;;
        esac
        """
    )


def render_specs_readme_template() -> str:
    return textwrap.dedent(
        """\
        # Specs

        This directory stores requirement specs.
        - Human-authored specs live at `specs/*.md`.
        - Auto-decomposed Ralph loop specs live at `specs/generated/*.md`.
        """
    )


def render_plan_block(
    documents: list[SourceDoc],
    tasks: list[Task],
    questions: list[Question],
    generated_spec_paths: list[str],
) -> str:
    lines: list[str] = [MANAGED_START, "## Ralph Loop Managed Plan", ""]
    lines.append("### Source Documents (Read-Only)")
    if documents:
        for doc in documents:
            lines.append(f"- `{doc.relpath}`")
    else:
        lines.append("- No source documents found. Add `prd.md`, `trd.md`, or `todo.md`.")
    lines.append("")

    lines.append("### Priority Queue")
    for index, task in enumerate(tasks, start=1):
        lines.append(f"- [ ] P{index}: {task.title}")
    lines.append("")

    lines.append("### Generated Specs")
    if generated_spec_paths:
        for spec_path in generated_spec_paths:
            lines.append(f"- `{spec_path}`")
    else:
        lines.append("- none")
    lines.append("")

    lines.append("### Task Cards")
    for index, task in enumerate(tasks, start=1):
        lines.append(f"#### T{index:02d} - {task.title}")
        lines.append(f"- Source: `{task.source}`")
        lines.append("- Done checklist:")
        lines.append("  - [ ] Behavior is implemented and mapped to source requirements.")
        lines.append("  - [ ] Acceptance criteria are explicit and testable.")
        lines.append("  - [ ] Tests are added or updated for this task.")
        lines.append("  - [ ] Verification commands in `AGENTS.md` pass.")
        lines.append("")

    lines.append("### Open Questions")
    if questions:
        for index, question in enumerate(questions, start=1):
            impacts = ", ".join(question.impacts)
            lines.append(
                f"- [ ] Q{index:02d}: {question.text} (impacts: {impacts}; source: `{question.source}`)"
            )
    else:
        lines.append("- [ ] None")
    lines.append("")

    lines.append("### Loop Rules")
    lines.append(
        "- Setup mode changes planning artifacts (`AGENTS.md`, `IMPLEMENTATION_PLAN.md`, `specs/generated/*.md`, `PROMPT_plan.md`, `PROMPT_build.md`, `loop.sh`, `.ralph-loop/*.py`)."
    )
    lines.append("- Build mode executes exactly one `[ ]` task per cycle.")
    lines.append("- Keep source docs (`prd.md`, `trd.md`, `todo.md`, `docs/*`, `README.md`) read-only during setup.")
    lines.append(MANAGED_END)
    return "\n".join(lines) + "\n"


def render_agents_block(verification_commands: list[str]) -> str:
    lines: list[str] = [MANAGED_START, "## Ralph Loop Contract", ""]
    lines.append("### Setup Mode (Default)")
    lines.append(
        "- Allowed writes: `AGENTS.md`, `IMPLEMENTATION_PLAN.md`, `specs/generated/*.md`, `PROMPT_plan.md`, `PROMPT_build.md`, `loop.sh`, `.ralph-loop/*.py`."
    )
    lines.append(
        "- Read-only inputs: `prd.md`, `trd.md`, `todo.md`, `docs/prd.md`, `docs/trd.md`, `docs/todo.md`, `specs/*.md` (human-authored), `README.md`."
    )
    lines.append("- Forbidden actions: source code edits, test execution, commits.")
    lines.append("")

    lines.append("### Build Mode (Explicit)")
    lines.append("- Select one highest-priority `[ ]` task and mark it `[-]`.")
    lines.append("- Implement only that task, then run verification commands.")
    lines.append("- Mark task `[x]` after checks pass.")
    lines.append("")

    lines.append("### Verification Commands")
    for command in verification_commands:
        lines.append(f"- `{command}`")
    lines.append(MANAGED_END)
    return "\n".join(lines) + "\n"


def upsert_managed_block(existing: str, managed_block: str) -> str:
    if MANAGED_START in existing and MANAGED_END in existing:
        before, after_start = existing.split(MANAGED_START, 1)
        _, after = after_start.split(MANAGED_END, 1)
        rebuilt = before.rstrip()
        if rebuilt:
            rebuilt += "\n\n"
        rebuilt += managed_block.strip() + "\n"
        remainder = after.lstrip("\n")
        if remainder:
            rebuilt += "\n" + remainder
        return rebuilt.rstrip() + "\n"

    base = existing.rstrip()
    if base:
        return base + "\n\n" + managed_block.strip() + "\n"
    return managed_block.strip() + "\n"


def initial_content(path: pathlib.Path, heading: str, template_path: pathlib.Path) -> str:
    if path.is_file():
        return read_text(path)
    if template_path.is_file():
        return read_text(template_path)
    return f"# {heading}\n"


def line_diff_summary(before: str, after: str) -> tuple[int, int]:
    added = 0
    removed = 0
    diff = difflib.unified_diff(before.splitlines(), after.splitlines(), lineterm="")
    for line in diff:
        if line.startswith("+++ ") or line.startswith("--- ") or line.startswith("@@"):
            continue
        if line.startswith("+"):
            added += 1
        elif line.startswith("-"):
            removed += 1
    return added, removed


def write_if_changed(path: pathlib.Path, content: str) -> tuple[str, int, int]:
    before = read_text(path) if path.is_file() else ""
    if before == content:
        return "unchanged", 0, 0
    path.write_text(content, encoding="utf-8")
    added, removed = line_diff_summary(before, content)
    state = "created" if not before else "updated"
    return state, added, removed


def apply_change(changes: list[FileChange], repo_root: pathlib.Path, path: pathlib.Path, content: str) -> None:
    state, added, removed = write_if_changed(path, content)
    rel = path.relative_to(repo_root).as_posix()
    changes.append(FileChange(path=rel, state=state, added=added, removed=removed))


def ensure_executable(path: pathlib.Path) -> None:
    if not path.exists():
        return
    mode = path.stat().st_mode
    path.chmod(mode | 0o111)


def write_generated_specs(
    repo_root: pathlib.Path,
    entries: list[SpecEntry],
    questions: list[Question],
    documents: list[SourceDoc],
    changes: list[FileChange],
) -> list[str]:
    specs_root = repo_root / "specs"
    generated_dir = specs_root / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)

    specs_readme = specs_root / "README.md"
    if not specs_readme.exists():
        apply_change(changes, repo_root, specs_readme, render_specs_readme_template())

    generated_paths: list[str] = []
    for entry in entries:
        path = generated_dir / f"{entry.slug}.md"
        apply_change(changes, repo_root, path, render_generated_spec(entry, questions))
        generated_paths.append(path.relative_to(repo_root).as_posix())

    index_path = generated_dir / "README.md"
    apply_change(changes, repo_root, index_path, render_generated_specs_index(entries, documents))
    # Keep plan listing order stable across full setup and --planning-only runs.
    return existing_generated_spec_paths(repo_root)


def render_runner_readme() -> str:
    return textwrap.dedent(
        """\
        # Ralph Loop Runtime

        This directory contains repository-local Ralph loop runtime scripts.
        Keep these files committed so loop execution remains portable across machines.
        """
    )


def write_repo_runner_files(
    repo_root: pathlib.Path,
    current_setup_script: pathlib.Path,
    current_next_task_script: pathlib.Path,
    changes: list[FileChange],
) -> None:
    runner_dir = repo_root / ".ralph-loop"
    runner_dir.mkdir(parents=True, exist_ok=True)

    repo_setup = runner_dir / "setup.py"
    repo_next_task = runner_dir / "next-task.py"
    repo_readme = runner_dir / "README.md"

    apply_change(changes, repo_root, repo_setup, read_text(current_setup_script))
    apply_change(changes, repo_root, repo_next_task, read_text(current_next_task_script))
    apply_change(changes, repo_root, repo_readme, render_runner_readme())
    ensure_executable(repo_setup)
    ensure_executable(repo_next_task)


def existing_generated_spec_paths(repo_root: pathlib.Path) -> list[str]:
    generated_dir = repo_root / "specs" / "generated"
    if not generated_dir.is_dir():
        return []
    paths = [path.relative_to(repo_root).as_posix() for path in sorted(generated_dir.glob("*.md"))]
    readme = generated_dir / "README.md"
    readme_rel = readme.relative_to(repo_root).as_posix()
    if readme_rel in paths:
        paths.remove(readme_rel)
        paths.insert(0, readme_rel)
    return paths


def unique_questions(
    extracted: Iterable[Question], generated: Iterable[Question], limit: int = 10
) -> list[Question]:
    merged: list[Question] = []
    seen: set[str] = set()
    for question in (*extracted, *generated):
        key = question.text.casefold()
        if key in seen:
            continue
        seen.add(key)
        merged.append(question)
        if len(merged) >= limit:
            break
    return merged


def print_plan_preview(
    docs: list[SourceDoc],
    tasks: list[Task],
    questions: list[Question],
    spec_entries: list[SpecEntry],
    planning_only: bool,
) -> None:
    print("setup mode: planning-only (no source code edits)")
    print("read-only inputs: prd/trd/todo/docs/*/specs/*.md/README.md")
    if docs:
        print("detected source documents:")
        for doc in docs:
            print(f"  - {doc.relpath}")
    else:
        print("detected source documents: none")
    print("planned output files:")
    print("  - AGENTS.md")
    print("  - IMPLEMENTATION_PLAN.md")
    if not planning_only:
        print("  - specs/generated/*.md")
        print("  - PROMPT_plan.md")
        print("  - PROMPT_build.md")
        print("  - loop.sh")
        print("  - .ralph-loop/setup.py")
        print("  - .ralph-loop/next-task.py")
    print(f"task candidates: {len(tasks)}")
    print(f"generated specs: {len(spec_entries)}")
    print(f"open questions: {len(questions)}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Set up Ralph Loop planning artifacts for the current repository."
    )
    parser.add_argument(
        "repo_root",
        nargs="?",
        default=".",
        help="Repository root (default: current directory)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print plan preview without writing files",
    )
    parser.add_argument(
        "--planning-only",
        action="store_true",
        help="Update only AGENTS.md and IMPLEMENTATION_PLAN.md",
    )
    args = parser.parse_args()

    requested_root = pathlib.Path(args.repo_root).resolve()
    git_root = find_git_root(requested_root)
    if git_root is None:
        print(
            "error: no git repository detected. run setup from a repository root or subdirectory.",
            file=sys.stderr,
        )
        return 2

    current_setup_script = pathlib.Path(__file__).resolve()
    current_next_task_script = current_setup_script.with_name("next-task.py")
    skill_dir = current_setup_script.parents[1]
    if not current_next_task_script.is_file():
        print(
            f"error: missing helper script: {current_next_task_script}",
            file=sys.stderr,
        )
        return 2
    docs = discover_documents(git_root)
    tasks = extract_tasks(docs)
    questions = unique_questions(extract_questions(docs), missing_document_questions(docs))
    spec_entries = compute_spec_entries(tasks)
    verification_commands = detect_verification_commands(git_root)

    print_plan_preview(docs, tasks, questions, spec_entries, args.planning_only)
    if args.dry_run:
        return 0

    agents_template = skill_dir / "assets" / "AGENTS.template.md"
    plan_template = skill_dir / "assets" / "IMPLEMENTATION_PLAN.template.md"

    agents_path = git_root / "AGENTS.md"
    plan_path = git_root / "IMPLEMENTATION_PLAN.md"
    prompt_plan_path = git_root / "PROMPT_plan.md"
    prompt_build_path = git_root / "PROMPT_build.md"
    loop_path = git_root / "loop.sh"

    agents_existing = initial_content(agents_path, "AGENTS.md", agents_template)
    plan_existing = initial_content(plan_path, "IMPLEMENTATION_PLAN.md", plan_template)

    changes: list[FileChange] = []

    updated_agents = upsert_managed_block(agents_existing, render_agents_block(verification_commands))
    apply_change(changes, git_root, agents_path, updated_agents)

    generated_specs: list[str] = []
    if not args.planning_only:
        generated_specs = write_generated_specs(git_root, spec_entries, questions, docs, changes)
        apply_change(changes, git_root, prompt_plan_path, render_prompt_plan())
        apply_change(changes, git_root, prompt_build_path, render_prompt_build())
        write_repo_runner_files(
            git_root,
            current_setup_script=current_setup_script,
            current_next_task_script=current_next_task_script,
            changes=changes,
        )
        apply_change(
            changes,
            git_root,
            loop_path,
            render_loop_script(),
        )
        ensure_executable(loop_path)
    else:
        generated_specs = existing_generated_spec_paths(git_root)

    updated_plan = upsert_managed_block(
        plan_existing, render_plan_block(docs, tasks, questions, generated_specs)
    )
    apply_change(changes, git_root, plan_path, updated_plan)

    print("change summary:")
    for change in changes:
        print(f"  - {change.path}: {change.state} (+{change.added}/-{change.removed})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
