from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from pipelines.validate_training_outputs import (
    CheckResult,
    check_backup_bundle,
    check_checkpoints,
    check_final_configs,
    check_scaler,
    check_scoring_smoke,
    check_split_policy,
    check_tensorboard_logs,
)

VALID_NODE_STATUS = {"idle", "running", "done", "fail"}
RUN_ID_ALLOWED_PATTERN = re.compile(r"[^a-zA-Z0-9._-]+")


def _resolve(repo_root: Path, maybe_relative: Path) -> Path:
    return maybe_relative if maybe_relative.is_absolute() else repo_root / maybe_relative


def _pick_first_existing(repo_root: Path, candidates: list[Path]) -> Path:
    for candidate in candidates:
        resolved = _resolve(repo_root, candidate)
        if resolved.exists():
            return resolved
    return _resolve(repo_root, candidates[0])


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _load_loss_series(path: Path) -> list[dict[str, float | int]]:
    if not path.exists():
        return []

    rows: list[dict[str, float | int]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                rows.append(
                    {
                        "epoch": int(row["epoch"]),
                        "train_loss": float(row["train_loss"]),
                        "val_loss": float(row["val_loss"]),
                    }
                )
            except Exception:
                continue
    return rows


def _count_event_files(path: Path) -> int:
    if not path.exists():
        return 0
    return len(list(path.rglob("events.out.tfevents*")))


def _find_backup_files(repo_root: Path) -> list[dict[str, Any]]:
    candidates: list[Path] = []
    candidates.extend(sorted((repo_root / "artifacts" / "bundles").glob("*.zip")))
    candidates.extend(sorted((repo_root / "artifacts" / "bundles").glob("*.tar.gz")))
    candidates.extend(sorted(repo_root.glob("run_bundle*.zip")))
    candidates.extend(sorted(repo_root.glob("train_bundle*.zip")))

    results: list[dict[str, Any]] = []
    for file_path in candidates:
        if file_path.exists() and file_path.is_file():
            results.append(
                {
                    "path": str(file_path.relative_to(repo_root)),
                    "size_bytes": int(file_path.stat().st_size),
                }
            )
    return results


def _config_summary(config_path: Path) -> dict[str, Any]:
    cfg = _read_yaml(config_path)
    data = cfg.get("data", {}) if isinstance(cfg.get("data"), dict) else {}
    training = cfg.get("training", {}) if isinstance(cfg.get("training"), dict) else {}
    model = cfg.get("model", {}) if isinstance(cfg.get("model"), dict) else {}
    device = cfg.get("device", {}) if isinstance(cfg.get("device"), dict) else {}

    return {
        "config_path": str(config_path),
        "source": data.get("source"),
        "fs": data.get("fs"),
        "lr": training.get("lr"),
        "epochs": training.get("epochs"),
        "mask_ratio": model.get("mask_ratio"),
        "amp": device.get("amp"),
    }


def _extract_layout_node_ids(layout_path: Path) -> set[str]:
    if not layout_path.exists():
        return set()
    try:
        layout = json.loads(layout_path.read_text(encoding="utf-8"))
    except Exception:
        return set()

    node_ids: set[str] = set()
    views = layout.get("views", [])
    if not isinstance(views, list):
        return node_ids
    for view in views:
        if not isinstance(view, dict):
            continue
        nodes = view.get("nodes", [])
        if not isinstance(nodes, list):
            continue
        for node in nodes:
            if isinstance(node, dict) and isinstance(node.get("id"), str):
                node_ids.add(node["id"])
    return node_ids


def normalize_run_id(run_id: str | None) -> str:
    if run_id is None:
        return ""
    normalized = RUN_ID_ALLOWED_PATTERN.sub("_", run_id.strip())
    normalized = normalized.strip("_")
    return normalized


def _final_val_loss(series: list[dict[str, Any]]) -> float | None:
    if not isinstance(series, list) or not series:
        return None
    for item in reversed(series):
        if not isinstance(item, dict):
            continue
        value = item.get("val_loss")
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _build_run_index_entry(payload: dict[str, Any], snapshot_file: str) -> dict[str, Any]:
    checklist = payload.get("checklist", [])
    passed = sum(1 for item in checklist if isinstance(item, dict) and bool(item.get("passed")))
    total = len(checklist) if isinstance(checklist, list) else 0
    patch_loss = _final_val_loss(payload.get("metrics", {}).get("patchtst", {}).get("loss", []))
    swin_loss = _final_val_loss(payload.get("metrics", {}).get("swinmae", {}).get("loss", []))

    return {
        "run_id": str(payload.get("meta", {}).get("run_id", "")),
        "file": snapshot_file,
        "timestamp": str(payload.get("meta", {}).get("timestamp", "")),
        "checklist": {
            "passed": passed,
            "total": total,
        },
        "final_val_loss": {
            "patchtst": patch_loss,
            "swinmae": swin_loss,
        },
    }


def _persist_run_history(
    *,
    repo_root: Path,
    payload: dict[str, Any],
    run_history_dir: Path,
    run_history_limit: int,
    timestamp: str,
) -> dict[str, str]:
    history_root = _resolve(repo_root, run_history_dir)
    history_root.mkdir(parents=True, exist_ok=True)

    run_id = str(payload.get("meta", {}).get("run_id", "")).strip()
    if not run_id:
        raise ValueError("run_id must not be empty when persisting run history.")

    snapshot_file = f"{run_id}.json"
    snapshot_path = history_root / snapshot_file
    snapshot_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")

    index_path = history_root / "index.json"
    existing_runs: list[dict[str, Any]] = []
    if index_path.exists():
        try:
            existing_index = json.loads(index_path.read_text(encoding="utf-8"))
            if isinstance(existing_index, dict) and isinstance(existing_index.get("runs"), list):
                existing_runs = [item for item in existing_index["runs"] if isinstance(item, dict)]
        except Exception:
            existing_runs = []

    new_entry = _build_run_index_entry(payload, snapshot_file=snapshot_file)
    merged_runs = [new_entry]
    merged_runs.extend(item for item in existing_runs if item.get("run_id") != run_id)

    limit = max(int(run_history_limit), 1)
    pruned = merged_runs[limit:]
    merged_runs = merged_runs[:limit]

    kept_files = {
        item.get("file")
        for item in merged_runs
        if isinstance(item.get("file"), str) and item.get("file")
    }
    for item in pruned:
        file_name = item.get("file")
        if not isinstance(file_name, str) or not file_name or file_name in kept_files:
            continue
        candidate = history_root / file_name
        if candidate.exists() and candidate.is_file():
            candidate.unlink()

    index_payload = {
        "generated_at": timestamp,
        "runs": merged_runs,
    }
    index_path.write_text(json.dumps(index_payload, indent=2, ensure_ascii=True), encoding="utf-8")

    return {
        "snapshot_path": str(snapshot_path.relative_to(repo_root)),
        "index_path": str(index_path.relative_to(repo_root)),
    }


def _collect_checklist_results(
    *,
    repo_root: Path,
    patch_ckpt: Path,
    swin_ckpt: Path,
    scaler_path: Path,
    runs_dir: Path,
    patch_cfg: Path,
    swin_cfg: Path,
    backup_path: Path | None,
    split_policy_path: Path,
    run_smoke: bool,
    smoke_timeout_sec: int,
) -> list[CheckResult]:
    return [
        check_checkpoints(1, patch_ckpt=patch_ckpt, swin_ckpt=swin_ckpt),
        check_scaler(2, scaler_path=scaler_path),
        check_tensorboard_logs(3, runs_dir=runs_dir),
        check_final_configs(4, patch_cfg_path=patch_cfg, swin_cfg_path=swin_cfg),
        check_backup_bundle(5, repo_root=repo_root, backup_path=backup_path),
        check_scoring_smoke(
            6,
            repo_root=repo_root,
            patch_ckpt=patch_ckpt,
            patch_cfg=patch_cfg,
            swin_ckpt=swin_ckpt,
            swin_cfg=swin_cfg,
            timeout_sec=smoke_timeout_sec,
            skip=not run_smoke,
        ),
        check_split_policy(7, split_policy_path=split_policy_path),
    ]


def _build_node_statuses(
    *,
    node_ids: set[str],
    checklist: list[CheckResult],
    artifacts: dict[str, Any],
    timestamp: str,
) -> dict[str, dict[str, str]]:
    result_map = {item.index: item for item in checklist}
    checkpoints_ready = bool(result_map[1].passed)
    scaler_ready = bool(result_map[2].passed)
    logs_ready = bool(result_map[3].passed)
    configs_ready = bool(result_map[4].passed)
    backup_ready = bool(result_map[5].passed)
    smoke_ready = bool(result_map[6].passed)
    split_ready = bool(result_map[7].passed)
    overall_ready = all(
        [
            checkpoints_ready,
            scaler_ready,
            logs_ready,
            configs_ready,
            backup_ready,
            smoke_ready,
            split_ready,
        ]
    )

    defaults = {
        "orchestrator",
        "data-prep",
        "dqvl",
        "patchtst",
        "swinmae",
        "artifact-save",
        "validation-gate",
        "run-summary",
        "checkpoint-check",
        "scaler-check",
        "runs-check",
        "config-check",
        "release-ready",
    }
    all_node_ids = set(node_ids) | defaults

    patch_exists = bool(artifacts["checkpoints"]["patchtst"]["exists"])
    swin_exists = bool(artifacts["checkpoints"]["swinmae"]["exists"])

    node_status: dict[str, dict[str, str]] = {}

    def set_status(node_id: str, done: bool, message: str) -> None:
        node_status[node_id] = {
            "status": "done" if done else "fail",
            "message": message,
            "updated_at": timestamp,
        }

    set_status(
        "orchestrator",
        checkpoints_ready or logs_ready or configs_ready,
        "Training run metadata collected.",
    )
    set_status("data-prep", split_ready, result_map[7].detail)
    set_status("dqvl", configs_ready, result_map[4].detail)
    set_status("patchtst", patch_exists, artifacts["checkpoints"]["patchtst"]["detail"])
    set_status("swinmae", swin_exists, artifacts["checkpoints"]["swinmae"]["detail"])
    set_status(
        "artifact-save",
        checkpoints_ready and scaler_ready and logs_ready,
        "Artifact readiness derived from checkpoint/scaler/log checks.",
    )
    set_status("validation-gate", overall_ready, "Checklist aggregate status.")
    set_status("run-summary", overall_ready, "Run summary available for dashboard.")
    set_status("checkpoint-check", checkpoints_ready, result_map[1].detail)
    set_status("scaler-check", scaler_ready, result_map[2].detail)
    set_status("runs-check", logs_ready, result_map[3].detail)
    set_status("config-check", configs_ready, result_map[4].detail)
    set_status("release-ready", overall_ready, "All required checks passed.")

    for node_id in all_node_ids:
        if node_id not in node_status:
            node_status[node_id] = {
                "status": "idle",
                "message": "No mapped runtime status yet.",
                "updated_at": timestamp,
            }

    return node_status


def validate_dashboard_state_schema(payload: dict[str, Any]) -> None:
    required_top = {"meta", "nodes", "checklist", "metrics", "artifacts"}
    missing_top = sorted(required_top - set(payload.keys()))
    if missing_top:
        raise ValueError(f"dashboard state missing top-level keys: {missing_top}")

    meta = payload["meta"]
    if not isinstance(meta, dict):
        raise ValueError("meta must be a mapping")
    for key in ["run_id", "timestamp", "repo_root"]:
        if key not in meta:
            raise ValueError(f"meta missing key: {key}")

    nodes = payload["nodes"]
    if not isinstance(nodes, dict):
        raise ValueError("nodes must be a mapping")
    for node_id, node in nodes.items():
        if not isinstance(node_id, str):
            raise ValueError("node id must be string")
        if not isinstance(node, dict):
            raise ValueError(f"node status must be mapping: {node_id}")
        for key in ["status", "message", "updated_at"]:
            if key not in node:
                raise ValueError(f"node {node_id} missing key: {key}")
        if node["status"] not in VALID_NODE_STATUS:
            raise ValueError(f"node {node_id} has invalid status: {node['status']}")

    checklist = payload["checklist"]
    if not isinstance(checklist, list):
        raise ValueError("checklist must be a list")
    for item in checklist:
        if not isinstance(item, dict):
            raise ValueError("checklist item must be mapping")
        for key in ["index", "title", "passed", "detail"]:
            if key not in item:
                raise ValueError(f"checklist item missing key: {key}")

    metrics = payload["metrics"]
    if not isinstance(metrics, dict):
        raise ValueError("metrics must be a mapping")
    for stream in ["patchtst", "swinmae"]:
        if stream not in metrics:
            raise ValueError(f"metrics missing stream: {stream}")
        stream_data = metrics[stream]
        if not isinstance(stream_data, dict):
            raise ValueError(f"metrics stream payload must be mapping: {stream}")
        for key in ["loss", "config"]:
            if key not in stream_data:
                raise ValueError(f"metrics.{stream} missing key: {key}")
        if not isinstance(stream_data["loss"], list):
            raise ValueError(f"metrics.{stream}.loss must be a list")
        if not isinstance(stream_data["config"], dict):
            raise ValueError(f"metrics.{stream}.config must be a mapping")

    artifacts = payload["artifacts"]
    if not isinstance(artifacts, dict):
        raise ValueError("artifacts must be a mapping")
    for key in ["readiness", "checkpoints", "scaler", "logs", "backup", "loss_files"]:
        if key not in artifacts:
            raise ValueError(f"artifacts missing key: {key}")


def export_dashboard_state(
    *,
    repo_root: Path,
    out_path: Path,
    run_id: str | None = None,
    patch_checkpoint: Path = Path("checkpoints/patchtst_ssl.pt"),
    swin_checkpoint: Path = Path("checkpoints/swinmae_ssl.pt"),
    scaler_path: Path = Path("artifacts/scaler_fdc.json"),
    runs_dir: Path = Path("runs"),
    patch_config: Path | None = None,
    swin_config: Path | None = None,
    backup_path: Path | None = None,
    split_policy_path: Path = Path("docs/calibration_split_policy.md"),
    layout_path: Path = Path("training_dashboard/data/dashboard-layout.json"),
    run_smoke: bool = False,
    smoke_timeout_sec: int = 300,
    persist_run_history: bool = False,
    run_history_dir: Path = Path("training_dashboard/data/runs"),
    run_history_limit: int = 20,
) -> dict[str, Any]:
    root = repo_root.resolve()
    timestamp = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    generated_run_id = datetime.now(timezone.utc).strftime("run-%Y%m%d-%H%M%S")
    effective_run_id = normalize_run_id(run_id) or generated_run_id

    patch_ckpt = _resolve(root, patch_checkpoint)
    swin_ckpt = _resolve(root, swin_checkpoint)
    scaler_file = _resolve(root, scaler_path)
    run_logs = _resolve(root, runs_dir)
    split_policy_file = _resolve(root, split_policy_path)
    layout_file = _resolve(root, layout_path)
    backup_file = _resolve(root, backup_path) if backup_path is not None else None

    patch_cfg = (
        _resolve(root, patch_config)
        if patch_config is not None
        else _pick_first_existing(
            root,
            [Path("configs/patchtst_ssl_real.yaml"), Path("configs/patchtst_ssl.yaml")],
        )
    )
    swin_cfg = (
        _resolve(root, swin_config)
        if swin_config is not None
        else _pick_first_existing(
            root,
            [Path("configs/swinmae_ssl_real.yaml"), Path("configs/swinmae_ssl.yaml")],
        )
    )

    checklist_results = _collect_checklist_results(
        repo_root=root,
        patch_ckpt=patch_ckpt,
        swin_ckpt=swin_ckpt,
        scaler_path=scaler_file,
        runs_dir=run_logs,
        patch_cfg=patch_cfg,
        swin_cfg=swin_cfg,
        backup_path=backup_file,
        split_policy_path=split_policy_file,
        run_smoke=run_smoke,
        smoke_timeout_sec=int(smoke_timeout_sec),
    )

    patch_loss_csv = root / "artifacts" / "loss" / "patchtst_loss_history.csv"
    patch_loss_png = root / "artifacts" / "loss" / "patchtst_loss_curve.png"
    swin_loss_csv = root / "artifacts" / "loss" / "swinmae_loss_history.csv"
    swin_loss_png = root / "artifacts" / "loss" / "swinmae_loss_curve.png"

    artifacts = {
        "readiness": {
            "checkpoints_ready": checklist_results[0].passed,
            "scaler_ready": checklist_results[1].passed,
            "logs_ready": checklist_results[2].passed,
            "configs_ready": checklist_results[3].passed,
            "backup_ready": checklist_results[4].passed,
            "smoke_ready": checklist_results[5].passed,
            "split_policy_ready": checklist_results[6].passed,
        },
        "checkpoints": {
            "patchtst": {
                "path": str(patch_ckpt),
                "exists": patch_ckpt.exists(),
                "size_bytes": int(patch_ckpt.stat().st_size) if patch_ckpt.exists() else 0,
                "detail": checklist_results[0].detail,
            },
            "swinmae": {
                "path": str(swin_ckpt),
                "exists": swin_ckpt.exists(),
                "size_bytes": int(swin_ckpt.stat().st_size) if swin_ckpt.exists() else 0,
                "detail": checklist_results[0].detail,
            },
        },
        "scaler": {
            "path": str(scaler_file),
            "exists": scaler_file.exists(),
            "size_bytes": int(scaler_file.stat().st_size) if scaler_file.exists() else 0,
            "detail": checklist_results[1].detail,
        },
        "logs": {
            "patchtst": {
                "path": str(run_logs / "patchtst_ssl"),
                "exists": (run_logs / "patchtst_ssl").exists(),
                "event_files": _count_event_files(run_logs / "patchtst_ssl"),
            },
            "swinmae": {
                "path": str(run_logs / "swinmae_ssl"),
                "exists": (run_logs / "swinmae_ssl").exists(),
                "event_files": _count_event_files(run_logs / "swinmae_ssl"),
            },
            "detail": checklist_results[2].detail,
        },
        "backup": {
            "found": checklist_results[4].passed,
            "detail": checklist_results[4].detail,
            "files": _find_backup_files(root),
        },
        "loss_files": {
            "patchtst": {
                "csv_path": str(patch_loss_csv),
                "csv_exists": patch_loss_csv.exists(),
                "png_path": str(patch_loss_png),
                "png_exists": patch_loss_png.exists(),
            },
            "swinmae": {
                "csv_path": str(swin_loss_csv),
                "csv_exists": swin_loss_csv.exists(),
                "png_path": str(swin_loss_png),
                "png_exists": swin_loss_png.exists(),
            },
        },
    }

    node_ids = _extract_layout_node_ids(layout_file)
    node_statuses = _build_node_statuses(
        node_ids=node_ids,
        checklist=checklist_results,
        artifacts=artifacts,
        timestamp=timestamp,
    )

    payload: dict[str, Any] = {
        "meta": {
            "run_id": effective_run_id,
            "timestamp": timestamp,
            "repo_root": str(root),
            "layout_path": str(layout_file),
        },
        "nodes": node_statuses,
        "checklist": [asdict(item) for item in checklist_results],
        "metrics": {
            "patchtst": {
                "loss": _load_loss_series(patch_loss_csv),
                "config": _config_summary(patch_cfg),
            },
            "swinmae": {
                "loss": _load_loss_series(swin_loss_csv),
                "config": _config_summary(swin_cfg),
            },
        },
        "artifacts": artifacts,
    }

    validate_dashboard_state_schema(payload)

    output = _resolve(root, out_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")

    if persist_run_history:
        history_files = _persist_run_history(
            repo_root=root,
            payload=payload,
            run_history_dir=run_history_dir,
            run_history_limit=int(run_history_limit),
            timestamp=timestamp,
        )
        payload["meta"]["run_history"] = history_files
        output.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")

    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export runtime dashboard state JSON for training observability.",
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--out", type=Path, default=Path("training_dashboard/data/dashboard-state.json"))
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--patch-checkpoint", type=Path, default=Path("checkpoints/patchtst_ssl.pt"))
    parser.add_argument("--swin-checkpoint", type=Path, default=Path("checkpoints/swinmae_ssl.pt"))
    parser.add_argument("--scaler-path", type=Path, default=Path("artifacts/scaler_fdc.json"))
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--patch-config", type=Path, default=None)
    parser.add_argument("--swin-config", type=Path, default=None)
    parser.add_argument("--backup-path", type=Path, default=None)
    parser.add_argument("--split-policy-path", type=Path, default=Path("docs/calibration_split_policy.md"))
    parser.add_argument(
        "--layout-path",
        type=Path,
        default=Path("training_dashboard/data/dashboard-layout.json"),
    )
    parser.add_argument("--run-smoke", action="store_true")
    parser.add_argument("--smoke-timeout-sec", type=int, default=300)
    parser.add_argument("--persist-run-history", action="store_true")
    parser.add_argument(
        "--run-history-dir",
        type=Path,
        default=Path("training_dashboard/data/runs"),
    )
    parser.add_argument("--run-history-limit", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    state = export_dashboard_state(
        repo_root=args.repo_root,
        out_path=args.out,
        run_id=args.run_id,
        patch_checkpoint=args.patch_checkpoint,
        swin_checkpoint=args.swin_checkpoint,
        scaler_path=args.scaler_path,
        runs_dir=args.runs_dir,
        patch_config=args.patch_config,
        swin_config=args.swin_config,
        backup_path=args.backup_path,
        split_policy_path=args.split_policy_path,
        layout_path=args.layout_path,
        run_smoke=bool(args.run_smoke),
        smoke_timeout_sec=int(args.smoke_timeout_sec),
        persist_run_history=bool(args.persist_run_history),
        run_history_dir=args.run_history_dir,
        run_history_limit=int(args.run_history_limit),
    )
    passed = sum(1 for item in state["checklist"] if item["passed"])
    total = len(state["checklist"])
    print(f"exported: {args.out}")
    print(f"checklist: {passed}/{total} passed")
    history = state.get("meta", {}).get("run_history", {})
    if isinstance(history, dict) and history:
        print(f"run_snapshot: {history.get('snapshot_path')}")
        print(f"run_index: {history.get('index_path')}")


if __name__ == "__main__":
    main()
