from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class CheckResult:
    index: int
    title: str
    passed: bool
    detail: str
    hint: str | None = None


def _human_size(num_bytes: int) -> str:
    value = float(num_bytes)
    units = ["B", "KB", "MB", "GB", "TB"]
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.1f}{unit}"
        value /= 1024.0
    return f"{num_bytes}B"


def _get_dotted(cfg: dict[str, Any], dotted_key: str) -> Any:
    current: Any = cfg
    for key in dotted_key.split("."):
        if not isinstance(current, dict) or key not in current:
            raise KeyError(dotted_key)
        current = current[key]
    return current


def _load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be mapping: {path}")
    return data


def check_checkpoints(index: int, patch_ckpt: Path, swin_ckpt: Path) -> CheckResult:
    missing: list[str] = []
    bad_size: list[str] = []
    details: list[str] = []

    for name, path in [("patchtst", patch_ckpt), ("swinmae", swin_ckpt)]:
        if not path.exists():
            missing.append(f"{name}={path}")
            continue
        size = path.stat().st_size
        if size <= 0:
            bad_size.append(f"{name}={path}")
            continue
        details.append(f"{name}:{path.name}({_human_size(size)})")

    passed = not missing and not bad_size
    detail = ", ".join(details) if passed else "; ".join(missing + bad_size)
    hint = (
        "Re-run training and confirm checkpoint_path in config."
        if not passed
        else None
    )
    return CheckResult(
        index=index,
        title="Check trained checkpoints",
        passed=passed,
        detail=detail or "No checkpoint details available.",
        hint=hint,
    )


def check_scaler(index: int, scaler_path: Path) -> CheckResult:
    if not scaler_path.exists():
        return CheckResult(
            index=index,
            title="Check PatchTST scaler artifact",
            passed=False,
            detail=f"missing:{scaler_path}",
            hint="Run PatchTST training; it should save artifacts/scaler_fdc.json.",
        )

    try:
        payload = json.loads(scaler_path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - defensive path
        return CheckResult(
            index=index,
            title="Check PatchTST scaler artifact",
            passed=False,
            detail=f"invalid_json:{scaler_path} ({exc})",
            hint="Inspect scaler file and regenerate with training.",
        )

    required = {"method", "center", "scale"}
    missing_keys = sorted(required - set(payload.keys()))
    passed = not missing_keys
    detail = (
        f"{scaler_path.name} keys={sorted(payload.keys())}"
        if passed
        else f"{scaler_path.name} missing_keys={missing_keys}"
    )
    return CheckResult(
        index=index,
        title="Check PatchTST scaler artifact",
        passed=passed,
        detail=detail,
        hint=None if passed else "Ensure scaler was saved after dataset build.",
    )


def check_tensorboard_logs(index: int, runs_dir: Path) -> CheckResult:
    stream_dirs = {
        "patchtst": runs_dir / "patchtst_ssl",
        "swinmae": runs_dir / "swinmae_ssl",
    }
    missing: list[str] = []
    details: list[str] = []

    for stream, stream_dir in stream_dirs.items():
        if not stream_dir.exists():
            missing.append(f"{stream}_dir_missing:{stream_dir}")
            continue
        events = list(stream_dir.rglob("events.out.tfevents*"))
        if not events:
            missing.append(f"{stream}_event_missing:{stream_dir}")
            continue
        details.append(f"{stream}:{len(events)} event file(s)")

    passed = not missing
    return CheckResult(
        index=index,
        title="Check TensorBoard logs",
        passed=passed,
        detail=", ".join(details) if passed else "; ".join(missing),
        hint=None if passed else "Check runs/<stream>_ssl and training execution logs.",
    )


def check_final_configs(index: int, patch_cfg_path: Path, swin_cfg_path: Path) -> CheckResult:
    cfg_specs = [
        (
            "patchtst",
            patch_cfg_path,
            [
                "data.source",
                "data.path",
                "training.lr",
                "training.epochs",
                "model.mask_ratio",
                "device.amp",
            ],
        ),
        (
            "swinmae",
            swin_cfg_path,
            [
                "data.source",
                "data.path",
                "data.fs",
                "training.lr",
                "training.epochs",
                "model.mask_ratio",
                "device.amp",
            ],
        ),
    ]

    failures: list[str] = []
    details: list[str] = []

    for stream, path, required_keys in cfg_specs:
        if not path.exists():
            failures.append(f"{stream}_config_missing:{path}")
            continue
        try:
            cfg = _load_yaml(path)
        except Exception as exc:  # pragma: no cover - defensive path
            failures.append(f"{stream}_config_invalid:{path} ({exc})")
            continue

        missing = [k for k in required_keys if _key_missing(cfg, k)]
        if missing:
            failures.append(f"{stream}_config_missing_keys:{missing}")
            continue

        details.append(
            f"{stream}:source={cfg['data']['source']}, lr={cfg['training']['lr']}, "
            f"epochs={cfg['training']['epochs']}, mask_ratio={cfg['model']['mask_ratio']}, "
            f"amp={cfg['device']['amp']}"
        )

    passed = not failures
    return CheckResult(
        index=index,
        title="Check final training configs",
        passed=passed,
        detail=", ".join(details) if passed else "; ".join(failures),
        hint=None if passed else "Freeze final *_real.yaml values before calibration.",
    )


def _key_missing(cfg: dict[str, Any], dotted_key: str) -> bool:
    try:
        _get_dotted(cfg, dotted_key)
    except KeyError:
        return True
    return False


def check_backup_bundle(index: int, repo_root: Path, backup_path: Path | None) -> CheckResult:
    candidates: list[Path] = []
    if backup_path is not None:
        candidates = [backup_path]
    else:
        candidates.extend(sorted((repo_root / "artifacts" / "bundles").glob("*.zip")))
        candidates.extend(sorted((repo_root / "artifacts" / "bundles").glob("*.tar.gz")))
        candidates.extend(sorted(repo_root.glob("run_bundle*.zip")))
        candidates.extend(sorted(repo_root.glob("train_bundle*.zip")))

    valid = [p for p in candidates if p.exists() and p.is_file() and p.stat().st_size > 0]
    passed = len(valid) > 0
    detail = ", ".join(f"{p.name}({_human_size(p.stat().st_size)})" for p in valid)
    if not passed:
        if backup_path is not None:
            detail = f"missing_or_empty:{backup_path}"
        else:
            detail = "No backup archive found (artifacts/bundles or run_bundle*.zip)"

    return CheckResult(
        index=index,
        title="Check backup bundle",
        passed=passed,
        detail=detail,
        hint=(
            "Create backup archive to local/Drive after training outputs are ready."
            if not passed
            else None
        ),
    )


def check_scoring_smoke(
    index: int,
    repo_root: Path,
    patch_ckpt: Path,
    patch_cfg: Path,
    swin_ckpt: Path,
    swin_cfg: Path,
    timeout_sec: int,
    skip: bool,
) -> CheckResult:
    if skip:
        return CheckResult(
            index=index,
            title="Run scoring smoke test for both streams",
            passed=True,
            detail="Skipped by --skip-smoke.",
        )

    runs = [
        ("patchtst", patch_ckpt, patch_cfg),
        ("swinmae", swin_ckpt, swin_cfg),
    ]
    failures: list[str] = []
    details: list[str] = []

    for stream, ckpt, cfg in runs:
        cmd = [
            sys.executable,
            "-m",
            "inference.run_scoring_example",
            "--stream",
            stream,
            "--checkpoint",
            str(ckpt),
            "--config",
            str(cfg),
        ]
        proc = subprocess.run(
            cmd,
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
        if proc.returncode != 0:
            tail = (proc.stdout + "\n" + proc.stderr).strip().splitlines()[-1:] or ["no_output"]
            failures.append(f"{stream}:rc={proc.returncode} tail={tail[0]}")
        else:
            details.append(f"{stream}:ok")

    passed = not failures
    return CheckResult(
        index=index,
        title="Run scoring smoke test for both streams",
        passed=passed,
        detail=", ".join(details) if passed else "; ".join(failures),
        hint=None if passed else "Run command manually to inspect full traceback.",
    )


def check_split_policy(index: int, split_policy_path: Path) -> CheckResult:
    if not split_policy_path.exists():
        return CheckResult(
            index=index,
            title="Check split policy documentation",
            passed=False,
            detail=f"missing:{split_policy_path}",
            hint="Document explicit train/calibration(normal) split criteria.",
        )

    text = split_policy_path.read_text(encoding="utf-8").lower()
    required_terms = ["train", "calibration", "normal"]
    missing_terms = [term for term in required_terms if term not in text]
    passed = not missing_terms

    detail = (
        f"{split_policy_path} contains split criteria."
        if passed
        else f"{split_policy_path} missing terms:{missing_terms}"
    )
    return CheckResult(
        index=index,
        title="Check split policy documentation",
        passed=passed,
        detail=detail,
        hint=None if passed else "Add train/calibration(normal) criteria and file lists.",
    )


def _print_results(results: list[CheckResult]) -> None:
    print("Training Output Validation Checklist")
    print("=" * 40)
    for result in results:
        mark = "[v]" if result.passed else "[ ]"
        state = "PASS" if result.passed else "FAIL"
        print(f"{result.index}. {mark} {state} - {result.title}")
        print(f"   detail: {result.detail}")
        if result.hint:
            print(f"   hint: {result.hint}")
        print("")

    total = len(results)
    passed = sum(1 for r in results if r.passed)
    failed = total - passed
    print(f"Summary: {passed}/{total} passed, {failed} failed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate post-training outputs with checklist + PASS/FAIL status.",
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--patch-checkpoint", type=Path, default=Path("checkpoints/patchtst_ssl.pt"))
    parser.add_argument("--swin-checkpoint", type=Path, default=Path("checkpoints/swinmae_ssl.pt"))
    parser.add_argument("--scaler-path", type=Path, default=Path("artifacts/scaler_fdc.json"))
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--patch-config", type=Path, default=Path("configs/patchtst_ssl_real.yaml"))
    parser.add_argument("--swin-config", type=Path, default=Path("configs/swinmae_ssl_real.yaml"))
    parser.add_argument("--backup-path", type=Path, default=None)
    parser.add_argument("--split-policy-path", type=Path, default=Path("docs/calibration_split_policy.md"))
    parser.add_argument("--skip-smoke", action="store_true")
    parser.add_argument("--smoke-timeout-sec", type=int, default=300)
    return parser.parse_args()


def _resolve(repo_root: Path, maybe_relative: Path) -> Path:
    return maybe_relative if maybe_relative.is_absolute() else repo_root / maybe_relative


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()

    patch_ckpt = _resolve(repo_root, args.patch_checkpoint)
    swin_ckpt = _resolve(repo_root, args.swin_checkpoint)
    scaler_path = _resolve(repo_root, args.scaler_path)
    runs_dir = _resolve(repo_root, args.runs_dir)
    patch_cfg = _resolve(repo_root, args.patch_config)
    swin_cfg = _resolve(repo_root, args.swin_config)
    split_policy_path = _resolve(repo_root, args.split_policy_path)
    backup_path = (
        _resolve(repo_root, args.backup_path) if args.backup_path is not None else None
    )

    results = [
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
            timeout_sec=int(args.smoke_timeout_sec),
            skip=bool(args.skip_smoke),
        ),
        check_split_policy(7, split_policy_path=split_policy_path),
    ]

    _print_results(results)
    if any(not r.passed for r in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

