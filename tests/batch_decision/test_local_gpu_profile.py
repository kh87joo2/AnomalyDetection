from __future__ import annotations

from pathlib import Path

import yaml

from batch_decision.runner import load_and_validate_request


def _load_yaml(path: Path) -> dict:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_local_gpu_profile_loads_with_repo_threshold_template() -> None:
    config_path = Path("configs/batch_decision_runtime_local_gpu.yaml")
    request = load_and_validate_request(config_path)

    assert request.run_id == "batch_decision_local_gpu_validation"
    assert request.stream == "dual"
    assert request.input_paths.patchtst == "runtime_inputs/fdc/test_fdc.csv"
    assert request.input_paths.swinmae == "runtime_inputs/vibration/test_vibration.csv"
    assert request.output_dir == "artifacts/batch_decision/local_gpu_validation"
    assert request.artifacts.thresholds_path.endswith(
        "artifacts/thresholds/batch_decision_thresholds.json"
    )


def test_local_gpu_profile_keeps_same_runtime_contract_as_colab() -> None:
    colab_cfg = _load_yaml(Path("configs/batch_decision_runtime_colab.yaml"))
    local_cfg = _load_yaml(Path("configs/batch_decision_runtime_local_gpu.yaml"))

    assert set(colab_cfg.keys()) == set(local_cfg.keys()) == {"environment", "run", "preprocess"}
    assert set(colab_cfg["run"].keys()) == set(local_cfg["run"].keys())
    assert set(colab_cfg["run"]["artifact_paths"].keys()) == set(
        local_cfg["run"]["artifact_paths"].keys()
    )
    assert set(colab_cfg["run"]["input_paths"].keys()) == set(local_cfg["run"]["input_paths"].keys())
    assert set(colab_cfg["preprocess"].keys()) == set(local_cfg["preprocess"].keys())
    assert colab_cfg["run"]["artifact_paths"] == local_cfg["run"]["artifact_paths"]
    assert colab_cfg["preprocess"] == local_cfg["preprocess"]
    assert local_cfg["environment"]["profile"] == "local_gpu"
    assert colab_cfg["environment"]["profile"] == "colab"


def test_runbook_mentions_local_gpu_migration_flow() -> None:
    runbook = Path("docs/runbook.md").read_text(encoding="utf-8")

    assert "Phase 3A Local GPU migration profile" in runbook
    assert "configs/batch_decision_runtime_local_gpu.yaml" in runbook
    assert "python3 -m batch_decision.runner --config configs/batch_decision_runtime_local_gpu.yaml --dry-run" in runbook
    assert "python3 -m batch_decision.runner --config configs/batch_decision_runtime_local_gpu.yaml --run" in runbook
