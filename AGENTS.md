# Repository Guidelines

## Project Structure & Module Organization
This repository is currently in bootstrap state (no source files yet). Keep contributions organized with this layout as code is added:
- `src/anomalydetection/`: core package code (models, pipelines, utilities).
- `tests/`: unit/integration tests mirroring `src/` modules.
- `data/`: local sample or synthetic data only (no sensitive/raw production data).
- `notebooks/`: exploratory analysis; move reusable logic into `src/`.
- `configs/`: YAML/JSON experiment and runtime configs.

Example: `src/anomalydetection/preprocessing/scaler.py` should have tests in `tests/preprocessing/test_scaler.py`.

## Build, Test, and Development Commands
Use a local virtual environment and keep commands reproducible:
- `python -m venv .venv && source .venv/bin/activate`: create/activate env.
- `pip install -r requirements.txt`: install runtime dependencies.
- `pip install -r requirements-dev.txt`: install lint/test/dev tools.
- `pytest -q`: run the full test suite.
- `pytest tests/<module> -q`: run a focused test target.
- `ruff check .` and `ruff format .`: lint and format before opening a PR.

If a Makefile is introduced later, keep these commands mirrored in `make test`, `make lint`, and `make format`.

## Coding Style & Naming Conventions
- Target Python 3.11+.
- Use 4-space indentation and type hints on public functions.
- Files/modules: `snake_case.py`; classes: `PascalCase`; functions/variables: `snake_case`; constants: `UPPER_SNAKE_CASE`.
- Keep functions small and single-purpose; avoid notebook-only logic in production code.

## Testing Guidelines
- Framework: `pytest` with test files named `test_<unit>.py`.
- Test behavior, not implementation details; prefer deterministic fixtures.
- Add regression tests for bug fixes.
- Minimum expectation: new features include tests; changed logic updates existing tests.

## Commit & Pull Request Guidelines
There is no commit history yet in this directory, so adopt this convention now:
- Commit style: `type(scope): short summary` (example: `feat(preprocessing): add robust z-score transformer`).
- Keep commits focused and atomic.
- PRs should include: purpose, key changes, test evidence (`pytest`/lint output), and linked issue (`Closes #<id>` when applicable).
- Include before/after metrics or screenshots for result-facing changes (plots, dashboards, reports).

## Security & Configuration Tips
- Never commit secrets, tokens, or raw sensitive datasets.
- Use `.env` for local secrets and provide `.env.example` with placeholder keys.
- Pin dependency versions and review updates for reproducibility.
