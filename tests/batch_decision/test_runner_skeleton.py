from __future__ import annotations

import json
from pathlib import Path

from batch_decision.runner import load_and_validate_request, main


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_runtime_config(path: Path, *, stream: str, patch_path: str | None, swin_path: str | None) -> None:
    lines = [
        "run:",
        "  run_id: test-run",
        f"  stream: {stream}",
        "  input_paths:",
        f"    patchtst: {patch_path if patch_path is not None else 'null'}",
        f"    swinmae: {swin_path if swin_path is not None else 'null'}",
        "  artifact_paths:",
        "    thresholds: thresholds.json",
        "    patchtst_checkpoint: checkpoints/patchtst_ssl.pt",
        "    swinmae_checkpoint: checkpoints/swinmae_ssl.pt",
        "    scaler: artifacts/scaler_fdc.json",
        "  output_dir: artifacts/batch_decision",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def test_load_and_validate_request_resolves_threshold_path(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "thresholds.json",
        {
            "patchtst": {"warn": 0.6, "anomaly": 0.8},
            "swinmae": {"warn": 0.6, "anomaly": 0.8},
            "dual": {"warn": 0.7, "anomaly": 0.9},
        },
    )
    cfg_path = tmp_path / "runtime.yaml"
    _write_runtime_config(
        cfg_path,
        stream="dual",
        patch_path="data/fdc.csv",
        swin_path="data/vibration.csv",
    )

    request = load_and_validate_request(cfg_path)
    assert request.stream == "dual"
    assert request.input_paths.patchtst == "data/fdc.csv"
    assert request.input_paths.swinmae == "data/vibration.csv"
    assert request.artifacts.thresholds_path == str((tmp_path / "thresholds.json").resolve())


def test_runner_dry_run_succeeds_for_valid_config(tmp_path: Path, capsys) -> None:
    _write_json(
        tmp_path / "thresholds.json",
        {"patchtst": {"warn": 0.5, "anomaly": 0.7}},
    )
    cfg_path = tmp_path / "runtime.yaml"
    _write_runtime_config(
        cfg_path,
        stream="patchtst",
        patch_path="data/fdc.csv",
        swin_path=None,
    )

    rc = main(["--config", str(cfg_path), "--dry-run"])
    captured = capsys.readouterr()
    assert rc == 0
    assert "dry-run validation passed" in captured.out


def test_runner_dry_run_fails_when_dual_stream_missing_input(
    tmp_path: Path, capsys
) -> None:
    _write_json(
        tmp_path / "thresholds.json",
        {"dual": {"warn": 0.7, "anomaly": 0.9}},
    )
    cfg_path = tmp_path / "runtime.yaml"
    _write_runtime_config(
        cfg_path,
        stream="dual",
        patch_path="data/fdc.csv",
        swin_path=None,
    )

    rc = main(["--config", str(cfg_path), "--dry-run"])
    captured = capsys.readouterr()
    assert rc == 1
    assert "requires both" in captured.err


def test_runner_requires_dry_run_flag(tmp_path: Path, capsys) -> None:
    _write_json(
        tmp_path / "thresholds.json",
        {"patchtst": {"warn": 0.5, "anomaly": 0.7}},
    )
    cfg_path = tmp_path / "runtime.yaml"
    _write_runtime_config(
        cfg_path,
        stream="patchtst",
        patch_path="data/fdc.csv",
        swin_path=None,
    )

    rc = main(["--config", str(cfg_path)])
    captured = capsys.readouterr()
    assert rc == 2
    assert "validation-only mode" in captured.err

