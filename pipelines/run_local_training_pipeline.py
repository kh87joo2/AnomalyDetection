from __future__ import annotations

import argparse
import glob
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class CommandStep:
    name: str
    cmd: list[str]


def _resolve(repo_root: Path, maybe_relative: Path) -> Path:
    return maybe_relative if maybe_relative.is_absolute() else repo_root / maybe_relative


def _default_run_id() -> str:
    return datetime.now(timezone.utc).strftime("local-run-%Y%m%d-%H%M%S")


def _quoted(cmd: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def _has_glob(pattern: str) -> bool:
    return any(token in pattern for token in ["*", "?", "["])


def _resolve_pattern(repo_root: Path, pattern: str) -> str:
    path = Path(pattern)
    if path.is_absolute():
        return str(path)
    return str(repo_root / path)


def _validate_data_path(repo_root: Path, source: str, data_path: str | None, stream_name: str) -> None:
    if source == "synthetic":
        return
    if not data_path:
        raise SystemExit(f"[{stream_name}] data.path is required when data.source={source}.")

    resolved_pattern = _resolve_pattern(repo_root, data_path)
    if _has_glob(data_path):
        matches = glob.glob(resolved_pattern, recursive=True)
        if not matches:
            raise SystemExit(
                f"[{stream_name}] no files matched data.path pattern: {data_path} (resolved: {resolved_pattern})"
            )
        return

    candidate = Path(resolved_pattern)
    if not candidate.exists():
        raise SystemExit(f"[{stream_name}] data.path does not exist: {candidate}")


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


def _prepare_runtime_config(
    *,
    repo_root: Path,
    base_config: Path,
    runtime_config: Path,
    stream_name: str,
    source_override: str | None,
    data_path_override: str | None,
) -> tuple[Path, dict[str, Any]]:
    base_cfg_path = _resolve(repo_root, base_config)
    runtime_cfg_path = _resolve(repo_root, runtime_config)
    cfg = _read_yaml(base_cfg_path)
    data_cfg = cfg.setdefault("data", {})
    if not isinstance(data_cfg, dict):
        raise SystemExit(f"[{stream_name}] data section must be a mapping in config: {base_cfg_path}")

    if source_override is not None:
        data_cfg["source"] = source_override
    if data_path_override is not None:
        data_cfg["path"] = data_path_override

    source = str(data_cfg.get("source", "synthetic")).lower()
    data_path = data_cfg.get("path")
    data_path_str = str(data_path) if data_path is not None else None
    _validate_data_path(repo_root, source=source, data_path=data_path_str, stream_name=stream_name)
    _write_yaml(runtime_cfg_path, cfg)
    return runtime_cfg_path, cfg


def build_command_steps(
    *,
    repo_root: Path,
    patch_config: Path,
    swin_config: Path,
    patch_checkpoint: Path,
    swin_checkpoint: Path,
    dashboard_out: Path,
    run_id: str,
    run_smoke: bool,
    persist_run_history: bool,
    run_history_limit: int,
    skip_patchtst: bool,
    skip_swinmae: bool,
    skip_scoring: bool,
    skip_validate: bool,
    validate_skip_smoke: bool,
    skip_export: bool,
) -> list[CommandStep]:
    root = repo_root.resolve()
    patch_cfg = _resolve(root, patch_config)
    swin_cfg = _resolve(root, swin_config)
    patch_ckpt = _resolve(root, patch_checkpoint)
    swin_ckpt = _resolve(root, swin_checkpoint)
    out_path = _resolve(root, dashboard_out)

    steps: list[CommandStep] = []

    if not skip_patchtst:
        steps.append(
            CommandStep(
                name="train_patchtst",
                cmd=[
                    sys.executable,
                    "-m",
                    "trainers.train_patchtst_ssl",
                    "--config",
                    str(patch_cfg),
                ],
            )
        )

    if not skip_swinmae:
        steps.append(
            CommandStep(
                name="train_swinmae",
                cmd=[
                    sys.executable,
                    "-m",
                    "trainers.train_swinmae_ssl",
                    "--config",
                    str(swin_cfg),
                ],
            )
        )

    if not skip_scoring:
        if not skip_patchtst:
            steps.append(
                CommandStep(
                    name="score_patchtst",
                    cmd=[
                        sys.executable,
                        "-m",
                        "inference.run_scoring_example",
                        "--stream",
                        "patchtst",
                        "--checkpoint",
                        str(patch_ckpt),
                        "--config",
                        str(patch_cfg),
                    ],
                )
            )

        if not skip_swinmae:
            steps.append(
                CommandStep(
                    name="score_swinmae",
                    cmd=[
                        sys.executable,
                        "-m",
                        "inference.run_scoring_example",
                        "--stream",
                        "swinmae",
                        "--checkpoint",
                        str(swin_ckpt),
                        "--config",
                        str(swin_cfg),
                    ],
                )
            )

    if not skip_validate:
        validate_cmd = [
            sys.executable,
            "-m",
            "pipelines.validate_training_outputs",
            "--repo-root",
            str(root),
            "--patch-config",
            str(patch_cfg),
            "--swin-config",
            str(swin_cfg),
        ]
        if validate_skip_smoke:
            validate_cmd.append("--skip-smoke")
        steps.append(CommandStep(name="validate_outputs", cmd=validate_cmd))

    if not skip_export:
        export_cmd = [
            sys.executable,
            "-m",
            "pipelines.export_training_dashboard_state",
            "--repo-root",
            str(root),
            "--out",
            str(out_path),
            "--run-id",
            run_id,
            "--patch-config",
            str(patch_cfg),
            "--swin-config",
            str(swin_cfg),
        ]
        if run_smoke:
            export_cmd.append("--run-smoke")
        if persist_run_history:
            export_cmd.extend(
                [
                    "--persist-run-history",
                    "--run-history-limit",
                    str(max(int(run_history_limit), 1)),
                ]
            )
        steps.append(CommandStep(name="export_dashboard_state", cmd=export_cmd))

    return steps


def _run_steps(repo_root: Path, steps: list[CommandStep], dry_run: bool) -> None:
    root = repo_root.resolve()

    for index, step in enumerate(steps, start=1):
        print(f"[{index}/{len(steps)}] {step.name}")
        print(f"  {_quoted(step.cmd)}")
        if dry_run:
            continue

        proc = subprocess.run(step.cmd, cwd=root)
        if proc.returncode != 0:
            raise SystemExit(proc.returncode)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Notebook-free local runner: train PatchTST/SwinMAE, score, validate, and export dashboard state."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--patch-config", type=Path, default=Path("configs/patchtst_ssl_local.yaml"))
    parser.add_argument("--swin-config", type=Path, default=Path("configs/swinmae_ssl_local.yaml"))
    parser.add_argument("--patch-checkpoint", type=Path, default=Path("checkpoints/patchtst_ssl.pt"))
    parser.add_argument("--swin-checkpoint", type=Path, default=Path("checkpoints/swinmae_ssl.pt"))
    parser.add_argument(
        "--dashboard-out",
        type=Path,
        default=Path("training_dashboard/data/dashboard-state.json"),
    )
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--run-smoke", action="store_true")
    parser.add_argument("--persist-run-history", action="store_true")
    parser.add_argument("--run-history-limit", type=int, default=20)
    parser.add_argument(
        "--patch-data-path",
        type=str,
        default=None,
        help="Local file path or glob for PatchTST input data. Example: data/fdc/*.csv",
    )
    parser.add_argument(
        "--swin-data-path",
        type=str,
        default=None,
        help="Local file path or glob for SwinMAE input data. Example: data/vib/*.csv",
    )
    parser.add_argument(
        "--patch-data-source",
        type=str,
        choices=["synthetic", "csv", "parquet"],
        default=None,
        help="Override PatchTST data.source in runtime config.",
    )
    parser.add_argument(
        "--swin-data-source",
        type=str,
        choices=["synthetic", "csv", "npy"],
        default=None,
        help="Override SwinMAE data.source in runtime config.",
    )
    parser.add_argument(
        "--runtime-config-dir",
        type=Path,
        default=Path("artifacts/runtime_configs"),
        help="Directory where generated runtime configs are written.",
    )

    parser.add_argument("--skip-patchtst", action="store_true")
    parser.add_argument("--skip-swinmae", action="store_true")
    parser.add_argument("--skip-scoring", action="store_true")
    parser.add_argument("--skip-validate", action="store_true")
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("--validate-skip-smoke", action="store_true")

    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()

    patch_cfg = _resolve(repo_root, args.patch_config)
    swin_cfg = _resolve(repo_root, args.swin_config)

    missing: list[Path] = []
    for cfg_path in [patch_cfg, swin_cfg]:
        if not cfg_path.exists():
            missing.append(cfg_path)

    if missing:
        joined = ", ".join(str(p) for p in missing)
        raise SystemExit(f"Missing config files: {joined}")

    patch_source = args.patch_data_source
    swin_source = args.swin_data_source
    if args.patch_data_path and patch_source is None:
        patch_source = "csv"
    if args.swin_data_path and swin_source is None:
        swin_source = "csv"

    runtime_dir = _resolve(repo_root, args.runtime_config_dir)
    patch_runtime_cfg, patch_cfg_payload = _prepare_runtime_config(
        repo_root=repo_root,
        base_config=args.patch_config,
        runtime_config=runtime_dir / "patchtst_runtime.yaml",
        stream_name="patchtst",
        source_override=patch_source,
        data_path_override=args.patch_data_path,
    )
    swin_runtime_cfg, swin_cfg_payload = _prepare_runtime_config(
        repo_root=repo_root,
        base_config=args.swin_config,
        runtime_config=runtime_dir / "swinmae_runtime.yaml",
        stream_name="swinmae",
        source_override=swin_source,
        data_path_override=args.swin_data_path,
    )

    patch_data = patch_cfg_payload.get("data", {}) if isinstance(patch_cfg_payload.get("data", {}), dict) else {}
    swin_data = swin_cfg_payload.get("data", {}) if isinstance(swin_cfg_payload.get("data", {}), dict) else {}
    print(
        f"[config] patchtst source={patch_data.get('source')} path={patch_data.get('path')} runtime={patch_runtime_cfg}"
    )
    print(
        f"[config] swinmae source={swin_data.get('source')} path={swin_data.get('path')} runtime={swin_runtime_cfg}"
    )

    effective_run_id = (args.run_id or "").strip() or _default_run_id()

    steps = build_command_steps(
        repo_root=repo_root,
        patch_config=patch_runtime_cfg,
        swin_config=swin_runtime_cfg,
        patch_checkpoint=args.patch_checkpoint,
        swin_checkpoint=args.swin_checkpoint,
        dashboard_out=args.dashboard_out,
        run_id=effective_run_id,
        run_smoke=bool(args.run_smoke),
        persist_run_history=bool(args.persist_run_history),
        run_history_limit=int(args.run_history_limit),
        skip_patchtst=bool(args.skip_patchtst),
        skip_swinmae=bool(args.skip_swinmae),
        skip_scoring=bool(args.skip_scoring),
        skip_validate=bool(args.skip_validate),
        validate_skip_smoke=bool(args.validate_skip_smoke),
        skip_export=bool(args.skip_export),
    )

    if not steps:
        raise SystemExit("No workflow steps selected. Remove --skip-* flags.")

    _run_steps(repo_root, steps=steps, dry_run=bool(args.dry_run))


if __name__ == "__main__":
    main()
