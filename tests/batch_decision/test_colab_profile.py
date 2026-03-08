from __future__ import annotations

from pathlib import Path

from batch_decision.runner import load_and_validate_request


def test_colab_profile_loads_with_repo_threshold_template() -> None:
    config_path = Path("configs/batch_decision_runtime_colab.yaml")
    request = load_and_validate_request(config_path)

    assert request.run_id == "batch_decision_colab_validation"
    assert request.stream == "dual"
    assert request.input_paths.patchtst == "data/fdc/test_fdc.csv"
    assert request.input_paths.swinmae == "data/vibration/test_vibration.csv"
    assert request.artifacts.thresholds_path.endswith(
        "artifacts/thresholds/batch_decision_thresholds.json"
    )


def test_runbook_mentions_colab_profile_flow() -> None:
    runbook = Path("docs/runbook.md").read_text(encoding="utf-8")

    assert "Phase 3A Batch Decision Colab profile" in runbook
    assert "configs/batch_decision_runtime_colab.yaml" in runbook
    assert "python3 -m batch_decision.runner --config configs/batch_decision_runtime_colab.yaml --dry-run" in runbook
    assert "python3 -m pytest -q tests/batch_decision/test_runner_skeleton.py tests/batch_decision/test_colab_profile.py" in runbook

