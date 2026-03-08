from __future__ import annotations

import argparse
import glob
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from batch_decision.decision_engine import decide_records, load_thresholds
from batch_decision.reporting import export_report
from batch_decision.runner import validate_runtime_config
from batch_decision.scoring_engine import score_batch_request

RUN_ID_ALLOWED_PATTERN = re.compile(r"[^a-zA-Z0-9._-]+")


def _resolve(repo_root: Path, maybe_relative: Path) -> Path:
    return maybe_relative if maybe_relative.is_absolute() else repo_root / maybe_relative


def _read_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"Config root must be a mapping: {path}")
    return payload


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )


def _resolve_pattern(repo_root: Path, pattern: str) -> str:
    path = Path(pattern)
    if path.is_absolute():
        return str(path)
    return str(repo_root / path)


def _validate_path(repo_root: Path, pattern: str | None, *, stream_name: str) -> None:
    if not pattern:
        raise SystemExit(f"[{stream_name}] test data path is required.")
    matches = glob.glob(_resolve_pattern(repo_root, pattern), recursive=True)
    if not matches:
        raise SystemExit(f"[{stream_name}] no files matched test data path: {pattern}")


def _default_run_id() -> str:
    return datetime.now(timezone.utc).strftime("local-test-%Y%m%d-%H%M%S")


def _normalize_run_id(run_id: str | None) -> str:
    if run_id is None:
        return ""
    normalized = RUN_ID_ALLOWED_PATTERN.sub("_", run_id.strip())
    return normalized.strip("_")


def _abspath_string(repo_root: Path, value: str | None) -> str | None:
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    return str(_resolve(repo_root, Path(text)))


def _prepare_runtime_config(
    *,
    repo_root: Path,
    base_config: Path,
    runtime_config: Path,
    run_id_override: str | None,
    stream_override: str | None,
    patch_test_path: str | None,
    swin_test_path: str | None,
    output_dir_override: str | None,
) -> tuple[Path, dict[str, Any]]:
    base_cfg_path = _resolve(repo_root, base_config)
    runtime_cfg_path = _resolve(repo_root, runtime_config)
    cfg = _read_yaml(base_cfg_path)

    run_cfg = cfg.setdefault("run", {})
    if not isinstance(run_cfg, dict):
        raise SystemExit(f"run section must be a mapping: {base_cfg_path}")

    effective_run_id = _normalize_run_id(run_id_override) or str(run_cfg.get("run_id", "")).strip() or _default_run_id()
    run_cfg["run_id"] = effective_run_id

    if stream_override is not None:
        run_cfg["stream"] = stream_override

    input_paths = run_cfg.setdefault("input_paths", {})
    if not isinstance(input_paths, dict):
        raise SystemExit(f"run.input_paths must be a mapping: {base_cfg_path}")

    if patch_test_path is not None:
        input_paths["patchtst"] = patch_test_path
    if swin_test_path is not None:
        input_paths["swinmae"] = swin_test_path
    if output_dir_override is not None:
        run_cfg["output_dir"] = output_dir_override

    if isinstance(input_paths.get("patchtst"), str):
        input_paths["patchtst"] = _resolve_pattern(repo_root, str(input_paths["patchtst"]))
    if isinstance(input_paths.get("swinmae"), str):
        input_paths["swinmae"] = _resolve_pattern(repo_root, str(input_paths["swinmae"]))

    artifact_paths = run_cfg.setdefault("artifact_paths", {})
    if not isinstance(artifact_paths, dict):
        raise SystemExit(f"run.artifact_paths must be a mapping: {base_cfg_path}")
    for key in ["thresholds", "patchtst_checkpoint", "swinmae_checkpoint", "scaler"]:
        current = artifact_paths.get(key)
        resolved = _abspath_string(repo_root, current if isinstance(current, str) else None)
        if resolved is not None:
            artifact_paths[key] = resolved

    preprocess_cfg = cfg.setdefault("preprocess", {})
    if not isinstance(preprocess_cfg, dict):
        raise SystemExit(f"preprocess section must be a mapping: {base_cfg_path}")
    for key in ["patchtst_config", "swinmae_config"]:
        current = preprocess_cfg.get(key)
        resolved = _abspath_string(repo_root, current if isinstance(current, str) else None)
        if resolved is not None:
            preprocess_cfg[key] = resolved

    stream = str(run_cfg.get("stream", "dual")).strip() or "dual"
    patch_path = input_paths.get("patchtst")
    swin_path = input_paths.get("swinmae")

    if stream in {"patchtst", "dual"}:
        _validate_path(repo_root, str(patch_path) if patch_path is not None else None, stream_name="patchtst")
    if stream in {"swinmae", "dual"}:
        _validate_path(repo_root, str(swin_path) if swin_path is not None else None, stream_name="swinmae")

    _write_yaml(runtime_cfg_path, cfg)
    return runtime_cfg_path, cfg


def _resolve_output_dir(repo_root: Path, output_dir: str | None, *, run_id: str, explicit_output_dir: str | None) -> Path:
    base_output = Path(output_dir or "artifacts/batch_decision/local_gpu")
    resolved = _resolve(repo_root, base_output)
    if explicit_output_dir is not None:
        return resolved
    normalized_run_id = _normalize_run_id(run_id) or _default_run_id()
    return resolved / normalized_run_id


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Local batch test runner: score test data and export decision results as JSON/CSV."
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--runtime-config",
        type=Path,
        default=Path("configs/batch_decision_runtime_local_gpu.yaml"),
    )
    parser.add_argument(
        "--runtime-config-dir",
        type=Path,
        default=Path("artifacts/runtime_configs"),
        help="Directory where the generated runtime config is written.",
    )
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--stream", choices=["patchtst", "swinmae", "dual"], default=None)
    parser.add_argument("--patch-test-path", type=str, default=None)
    parser.add_argument("--swin-test-path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = args.repo_root.resolve()
    runtime_dir = _resolve(repo_root, args.runtime_config_dir)
    runtime_cfg_path, runtime_cfg = _prepare_runtime_config(
        repo_root=repo_root,
        base_config=args.runtime_config,
        runtime_config=runtime_dir / "batch_decision_runtime_local_gpu.yaml",
        run_id_override=args.run_id,
        stream_override=args.stream,
        patch_test_path=args.patch_test_path,
        swin_test_path=args.swin_test_path,
        output_dir_override=args.output_dir,
    )

    request = validate_runtime_config(runtime_cfg, config_path=runtime_cfg_path)
    actual_output_dir = _resolve_output_dir(
        repo_root,
        request.output_dir,
        run_id=request.run_id,
        explicit_output_dir=args.output_dir,
    )

    print(f"[config] run_id={request.run_id}")
    print(f"[config] stream={request.stream}")
    print(f"[config] runtime={runtime_cfg_path}")
    print(f"[config] thresholds={request.artifacts.thresholds_path}")
    print(f"[config] output_dir={actual_output_dir}")

    if args.dry_run:
        return 0

    score_payload = score_batch_request(
        request,
        runtime_config=runtime_cfg,
        runtime_config_path=runtime_cfg_path,
    )

    streams: dict[str, Any] = {}
    if score_payload.patchtst_records:
        thresholds = load_thresholds(request.artifacts.thresholds_path, stream="patchtst")
        streams["patchtst"] = decide_records(score_payload.patchtst_records, thresholds=thresholds)
    if score_payload.swinmae_records:
        thresholds = load_thresholds(request.artifacts.thresholds_path, stream="swinmae")
        streams["swinmae"] = decide_records(score_payload.swinmae_records, thresholds=thresholds)
    if not streams:
        raise SystemExit("No scored records were produced. Check test data and model artifacts.")

    report_payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_id": request.run_id,
        "stream": request.stream,
        "thresholds_path": request.artifacts.thresholds_path,
        "score_metadata": score_payload.metadata,
        "output_dir": str(actual_output_dir),
        "streams": streams,
    }
    manifest = export_report(report_payload, output_dir=str(actual_output_dir))

    print(f"[report] summary_json={manifest['summary_json']}")
    for stream_name, info in manifest["streams"].items():
        payload = streams.get(stream_name, {})
        summary = payload.get("summary", {}) if isinstance(payload, dict) else {}
        decision_counts = summary.get("decision_counts", {}) if isinstance(summary, dict) else {}
        print(f"[report] {stream_name}_json={info['json']}")
        print(f"[report] {stream_name}_csv={info['csv']}")
        print(f"[report] {stream_name}_records={info['record_count']}")
        print(f"[report] {stream_name}_decisions={decision_counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
