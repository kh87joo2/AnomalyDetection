#!/usr/bin/env python3
import argparse
import pathlib
import re
import sys


TASK_PATTERN = re.compile(r"^\s*[-*]\s+\[ \]\s+(.+?)\s*$")


def first_open_task(plan_path: pathlib.Path) -> str | None:
    for line in plan_path.read_text(encoding="utf-8").splitlines():
        match = TASK_PATTERN.match(line)
        if match:
            return match.group(1)
    return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Print the first open task from IMPLEMENTATION_PLAN.md"
    )
    parser.add_argument(
        "plan",
        nargs="?",
        default="IMPLEMENTATION_PLAN.md",
        help="Path to plan file (default: IMPLEMENTATION_PLAN.md)",
    )
    args = parser.parse_args()

    plan_path = pathlib.Path(args.plan)
    if not plan_path.exists():
        print(f"plan not found: {plan_path}", file=sys.stderr)
        return 2

    task = first_open_task(plan_path)
    if task is None:
        print("no open task found")
        return 1

    print(task)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
