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
    eval "$RALPH_PLAN_CMD \"$REPO_ROOT/PROMPT_plan.md\""
  else
    echo "plan hook not set. Set RALPH_PLAN_CMD to invoke your coding agent with PROMPT_plan.md."
  fi
}

run_build_hook() {
  if [[ -n "${RALPH_BUILD_CMD:-}" ]]; then
    eval "$RALPH_BUILD_CMD \"$REPO_ROOT/PROMPT_build.md\""
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
