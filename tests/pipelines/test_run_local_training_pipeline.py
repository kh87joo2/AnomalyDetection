from __future__ import annotations

from pathlib import Path

import pytest

from pipelines.run_local_training_pipeline import (
    _prepare_runtime_config,
    build_command_steps,
)


def _touch(path: Path, content: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_build_command_steps_default_includes_train_validate_export(tmp_path: Path) -> None:
    _touch(tmp_path / "configs/patchtst_ssl_local.yaml", "seed: 42\n")
    _touch(tmp_path / "configs/swinmae_ssl_local.yaml", "seed: 42\n")

    steps = build_command_steps(
        repo_root=tmp_path,
        patch_config=Path("configs/patchtst_ssl_local.yaml"),
        swin_config=Path("configs/swinmae_ssl_local.yaml"),
        patch_checkpoint=Path("checkpoints/patchtst_ssl.pt"),
        swin_checkpoint=Path("checkpoints/swinmae_ssl.pt"),
        dashboard_out=Path("training_dashboard/data/dashboard-state.json"),
        run_id="local_run_001",
        run_smoke=False,
        persist_run_history=True,
        run_history_limit=20,
        skip_patchtst=False,
        skip_swinmae=False,
        skip_scoring=False,
        skip_validate=False,
        validate_skip_smoke=True,
        skip_export=False,
    )

    names = [step.name for step in steps]
    assert names == [
        "train_patchtst",
        "train_swinmae",
        "score_patchtst",
        "score_swinmae",
        "validate_outputs",
        "export_dashboard_state",
    ]

    validate_cmd = [step.cmd for step in steps if step.name == "validate_outputs"][0]
    assert "--skip-smoke" in validate_cmd

    export_cmd = [step.cmd for step in steps if step.name == "export_dashboard_state"][0]
    assert "--persist-run-history" in export_cmd
    assert "local_run_001" in export_cmd


def test_build_command_steps_skip_flags(tmp_path: Path) -> None:
    _touch(tmp_path / "configs/patchtst_ssl_local.yaml", "seed: 42\n")
    _touch(tmp_path / "configs/swinmae_ssl_local.yaml", "seed: 42\n")

    steps = build_command_steps(
        repo_root=tmp_path,
        patch_config=Path("configs/patchtst_ssl_local.yaml"),
        swin_config=Path("configs/swinmae_ssl_local.yaml"),
        patch_checkpoint=Path("checkpoints/patchtst_ssl.pt"),
        swin_checkpoint=Path("checkpoints/swinmae_ssl.pt"),
        dashboard_out=Path("training_dashboard/data/dashboard-state.json"),
        run_id="local_run_002",
        run_smoke=False,
        persist_run_history=False,
        run_history_limit=20,
        skip_patchtst=True,
        skip_swinmae=False,
        skip_scoring=False,
        skip_validate=False,
        validate_skip_smoke=False,
        skip_export=True,
    )

    names = [step.name for step in steps]
    assert "train_patchtst" not in names
    assert "score_patchtst" not in names
    assert "train_swinmae" in names
    assert "score_swinmae" in names
    assert "validate_outputs" in names
    assert "export_dashboard_state" not in names


def test_prepare_runtime_config_applies_local_data_overrides(tmp_path: Path) -> None:
    _touch(
        tmp_path / "configs/patchtst_ssl_local.yaml",
        "\n".join(
            [
                "seed: 42",
                "data:",
                "  source: synthetic",
                "  path: data/fdc/*.csv",
                "",
            ]
        ),
    )
    _touch(tmp_path / "data/fdc/local_patch.csv", "timestamp,a,b\n1,0.1,0.2\n")

    runtime_cfg_path, runtime_payload = _prepare_runtime_config(
        repo_root=tmp_path,
        base_config=Path("configs/patchtst_ssl_local.yaml"),
        runtime_config=Path("artifacts/runtime_configs/patchtst_runtime.yaml"),
        stream_name="patchtst",
        source_override="csv",
        data_path_override="data/fdc/*.csv",
    )

    assert runtime_cfg_path.exists()
    data_cfg = runtime_payload.get("data", {})
    assert data_cfg["source"] == "csv"
    assert data_cfg["path"] == "data/fdc/*.csv"


def test_prepare_runtime_config_raises_when_real_data_path_missing(tmp_path: Path) -> None:
    _touch(
        tmp_path / "configs/swinmae_ssl_local.yaml",
        "\n".join(
            [
                "seed: 42",
                "data:",
                "  source: csv",
                "  path: data/vib/*.csv",
                "",
            ]
        ),
    )

    with pytest.raises(SystemExit, match="no files matched data.path pattern"):
        _prepare_runtime_config(
            repo_root=tmp_path,
            base_config=Path("configs/swinmae_ssl_local.yaml"),
            runtime_config=Path("artifacts/runtime_configs/swinmae_runtime.yaml"),
            stream_name="swinmae",
            source_override="csv",
            data_path_override="data/vib/*.csv",
        )
