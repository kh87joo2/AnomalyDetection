from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, cast

from core.config import ConfigError, get_required, load_yaml_config

from batch_decision.contracts import ArtifactPaths, BatchRunRequest, InputPaths, StreamName

ALLOWED_STREAMS: set[str] = {"patchtst", "swinmae", "dual"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch anomaly decision runner (P0B skeleton)."
    )
    parser.add_argument("--config", type=Path, required=True, help="Runtime config YAML path.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config/contracts only (no scoring execution).",
    )
    parser.add_argument(
        "--score-only",
        action="store_true",
        help="Run preprocess + batch scoring only (no decision/report export yet).",
    )
    return parser.parse_args(argv)


def _as_mapping(value: Any, key_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"Config key must be a mapping: {key_name}")
    return value


def _require_str(mapping: dict[str, Any], key: str, context: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"Missing required non-empty string: {context}.{key}")
    return value.strip()


def _optional_str(mapping: dict[str, Any], key: str) -> str | None:
    value = mapping.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ConfigError(f"Config value must be string or null: {key}")
    value = value.strip()
    return value or None


def _resolve_path(raw_path: str, config_path: Path | None) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    candidates: list[Path] = []
    if config_path is not None:
        candidates.append((config_path.parent / path).resolve())
    candidates.append((Path.cwd() / path).resolve())
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _validate_input_paths(stream: str, input_paths: InputPaths) -> None:
    if stream == "patchtst" and not input_paths.patchtst:
        raise ConfigError("stream=patchtst requires run.input_paths.patchtst")
    if stream == "swinmae" and not input_paths.swinmae:
        raise ConfigError("stream=swinmae requires run.input_paths.swinmae")
    if stream == "dual" and (not input_paths.patchtst or not input_paths.swinmae):
        raise ConfigError(
            "stream=dual requires both run.input_paths.patchtst and run.input_paths.swinmae"
        )


def _validate_threshold_payload(
    payload: dict[str, Any], stream: str, thresholds_path: Path
) -> None:
    if stream not in payload:
        raise ConfigError(
            f"Threshold file must contain '{stream}' key: {thresholds_path}"
        )
    stream_thresholds = payload.get(stream)
    if not isinstance(stream_thresholds, dict):
        raise ConfigError(
            f"Threshold section '{stream}' must be a mapping: {thresholds_path}"
        )
    warn = stream_thresholds.get("warn")
    anomaly = stream_thresholds.get("anomaly")
    if not isinstance(warn, (int, float)) or not isinstance(anomaly, (int, float)):
        raise ConfigError(
            f"Threshold section '{stream}' must include numeric warn/anomaly values: {thresholds_path}"
        )
    if float(warn) >= float(anomaly):
        raise ConfigError(
            f"Threshold rule invalid for '{stream}': warn must be < anomaly ({thresholds_path})"
        )


def validate_runtime_config(
    config: dict[str, Any], *, config_path: Path | None = None
) -> BatchRunRequest:
    run_cfg = _as_mapping(get_required(config, "run"), "run")
    run_id = _require_str(run_cfg, "run_id", "run")
    stream = _require_str(run_cfg, "stream", "run")
    if stream not in ALLOWED_STREAMS:
        raise ConfigError(
            f"run.stream must be one of {sorted(ALLOWED_STREAMS)} (got: {stream})"
        )
    stream_name = cast(StreamName, stream)

    input_cfg = _as_mapping(get_required(config, "run.input_paths"), "run.input_paths")
    input_paths = InputPaths(
        patchtst=_optional_str(input_cfg, "patchtst"),
        swinmae=_optional_str(input_cfg, "swinmae"),
    )
    _validate_input_paths(stream=stream_name, input_paths=input_paths)

    artifact_cfg = _as_mapping(
        get_required(config, "run.artifact_paths"), "run.artifact_paths"
    )
    thresholds_raw = _require_str(artifact_cfg, "thresholds", "run.artifact_paths")
    thresholds_path = _resolve_path(thresholds_raw, config_path=config_path)
    if not thresholds_path.exists():
        raise ConfigError(f"Threshold file not found: {thresholds_path}")

    try:
        thresholds_payload = json.loads(thresholds_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError(f"Threshold file JSON decode failed: {thresholds_path} ({exc})") from exc
    if not isinstance(thresholds_payload, dict):
        raise ConfigError(f"Threshold file root must be a mapping: {thresholds_path}")
    _validate_threshold_payload(
        payload=thresholds_payload, stream=stream_name, thresholds_path=thresholds_path
    )

    output_dir = _optional_str(run_cfg, "output_dir")

    artifacts = ArtifactPaths(
        thresholds_path=str(thresholds_path),
        patchtst_checkpoint=_optional_str(artifact_cfg, "patchtst_checkpoint"),
        swinmae_checkpoint=_optional_str(artifact_cfg, "swinmae_checkpoint"),
        scaler_path=_optional_str(artifact_cfg, "scaler"),
    )
    return BatchRunRequest(
        run_id=run_id,
        stream=stream_name,
        input_paths=input_paths,
        artifacts=artifacts,
        output_dir=output_dir,
    )


def load_and_validate_request(config_path: Path) -> BatchRunRequest:
    cfg = load_yaml_config(config_path)
    return validate_runtime_config(cfg, config_path=config_path)


def run_dry_run(config_path: Path) -> BatchRunRequest:
    request = load_and_validate_request(config_path.resolve())
    print("batch_decision dry-run validation passed")
    print(f"run_id={request.run_id}")
    print(f"stream={request.stream}")
    print(f"thresholds={request.artifacts.thresholds_path}")
    return request


def run_score_only(config_path: Path) -> BatchRunRequest:
    runtime_config = load_yaml_config(config_path.resolve())
    request = validate_runtime_config(runtime_config, config_path=config_path.resolve())

    from batch_decision.scoring_engine import score_batch_request

    payload = score_batch_request(
        request,
        runtime_config=runtime_config,
        runtime_config_path=config_path.resolve(),
    )

    print("batch_decision score-only run completed")
    print(f"run_id={payload.run_id}")
    print(f"stream={payload.stream}")
    if payload.patchtst_records:
        print(f"patchtst_windows={len(payload.patchtst_records)}")
        print(
            "patchtst_score_sample="
            f"{[round(record.score, 6) for record in payload.patchtst_records[:5]]}"
        )
    if payload.swinmae_records:
        print(f"swinmae_windows={len(payload.swinmae_records)}")
        print(
            "swinmae_score_sample="
            f"{[round(record.score, 6) for record in payload.swinmae_records[:5]]}"
        )
    return request


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.dry_run and args.score_only:
        print("Choose exactly one mode: --dry-run or --score-only.", file=sys.stderr)
        return 2
    if not args.dry_run and not args.score_only:
        print(
            "P0B skeleton currently supports validation-only mode. Use --dry-run or --score-only.",
            file=sys.stderr,
        )
        return 2
    try:
        if args.dry_run:
            run_dry_run(config_path=args.config)
        else:
            run_score_only(config_path=args.config)
    except (ConfigError, FileNotFoundError, ValueError, RuntimeError) as exc:
        prefix = "config validation failed" if args.dry_run else "score-only failed"
        print(f"{prefix}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
