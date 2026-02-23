from __future__ import annotations

import argparse
import cgi
import json
import re
import shutil
import subprocess
import sys
import threading
import zipfile
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

REPO_ROOT = Path(__file__).resolve().parents[1]
DASHBOARD_ROOT = REPO_ROOT / "training_dashboard"
UPLOAD_ROOT = DASHBOARD_ROOT / "uploads"

STREAM_SPECS: dict[str, dict[str, Any]] = {
    "patchtst": {
        "dir": UPLOAD_ROOT / "patchtst",
        "allowed_suffixes": {".csv", ".parquet", ".pq", ".zip"},
    },
    "swinmae": {
        "dir": UPLOAD_ROOT / "swinmae",
        "allowed_suffixes": {".csv", ".npy", ".zip"},
    },
}

PIPELINE_STEP_PATTERN = re.compile(r"^\[(\d+)/(\d+)\]\s+([a-z0-9_]+)\s*$")
PATCH_EPOCH_PATTERN = re.compile(r"\[PatchTST\]\[Epoch\s+\d+\]")
SWIN_EPOCH_PATTERN = re.compile(r"\[SwinMAE\]\[Epoch\s+\d+\]")

# Keep this aligned with training_dashboard/data/dashboard-layout.json node IDs.
DASHBOARD_NODE_IDS: tuple[str, ...] = (
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
)

STEP_INITIAL_NODES: dict[str, tuple[str, ...]] = {
    "train_patchtst": ("data-prep", "dqvl"),
    "train_swinmae": ("swinmae",),
    "score_patchtst": ("artifact-save",),
    "score_swinmae": ("artifact-save",),
    "validate_outputs": ("validation-gate",),
    "export_dashboard_state": ("run-summary",),
}

STEP_ACTIVE_NODES_ON_EPOCH: dict[str, tuple[re.Pattern[str], tuple[str, ...]]] = {
    "train_patchtst": (PATCH_EPOCH_PATTERN, ("patchtst",)),
    "train_swinmae": (SWIN_EPOCH_PATTERN, ("swinmae",)),
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _safe_filename(raw_name: str) -> str:
    base = Path(raw_name).name
    cleaned = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in base)
    return cleaned or "upload.bin"


def _unique_path(path: Path) -> Path:
    if not path.exists():
        return path

    for index in range(1, 10_000):
        candidate = path.with_name(f"{path.stem}_{index}{path.suffix}")
        if not candidate.exists():
            return candidate

    raise RuntimeError(f"Unable to allocate unique path for {path}")


def _stream_dir(stream: str) -> Path:
    return STREAM_SPECS[stream]["dir"]


def _normalize_saved_name(stream: str, filename: str) -> str:
    safe = _safe_filename(filename)
    suffix = Path(safe).suffix.lower()
    if stream == "patchtst" and suffix == ".pq":
        return str(Path(safe).with_suffix(".parquet"))
    return safe


def _extract_zip_flat(stream: str, zip_path: Path) -> list[Path]:
    stream_dir = _stream_dir(stream)
    saved_paths: list[Path] = []

    with zipfile.ZipFile(zip_path, "r") as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue

            member_name = _normalize_saved_name(stream, Path(info.filename).name)
            suffix = Path(member_name).suffix.lower()
            allowed = STREAM_SPECS[stream]["allowed_suffixes"]
            if suffix not in allowed or suffix == ".zip":
                continue

            target = _unique_path(stream_dir / member_name)
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info, "r") as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            saved_paths.append(target)

    if not saved_paths:
        raise ValueError("Zip file did not contain supported data files.")

    return saved_paths


def save_uploaded_blob(stream: str, filename: str, blob: bytes) -> list[Path]:
    stream_dir = _stream_dir(stream)
    stream_dir.mkdir(parents=True, exist_ok=True)

    normalized_name = _normalize_saved_name(stream, filename)
    suffix = Path(normalized_name).suffix.lower()
    allowed = STREAM_SPECS[stream]["allowed_suffixes"]
    if suffix not in allowed:
        raise ValueError(f"Unsupported file type for {stream}: {suffix or '<none>'}")

    if suffix == ".zip":
        zip_target = _unique_path(stream_dir / normalized_name)
        zip_target.write_bytes(blob)
        try:
            extracted = _extract_zip_flat(stream, zip_target)
        finally:
            if zip_target.exists():
                zip_target.unlink()
        return extracted

    target = _unique_path(stream_dir / normalized_name)
    target.write_bytes(blob)
    return [target]


def list_stream_files(stream: str) -> list[Path]:
    stream_dir = _stream_dir(stream)
    if not stream_dir.exists():
        return []
    return sorted([p for p in stream_dir.iterdir() if p.is_file()])


def summarize_uploads() -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for stream in STREAM_SPECS:
        files = list_stream_files(stream)
        payload[stream] = {
            "count": len(files),
            "files": [
                {
                    "name": f.name,
                    "path": str(f.relative_to(REPO_ROOT)),
                    "size_bytes": int(f.stat().st_size),
                }
                for f in files
            ],
        }
    return payload


def infer_stream_runtime(stream: str) -> tuple[str, str]:
    files = list_stream_files(stream)
    if not files:
        raise ValueError(f"No files uploaded for {stream}.")

    suffixes = {path.suffix.lower() for path in files}

    if stream == "patchtst":
        if suffixes == {".csv"}:
            return "csv", str((_stream_dir(stream) / "*.csv").resolve())
        if suffixes <= {".parquet"}:
            return "parquet", str((_stream_dir(stream) / "*.parquet").resolve())
        raise ValueError("PatchTST files must be all CSV or all Parquet.")

    if stream == "swinmae":
        if suffixes == {".csv"}:
            return "csv", str((_stream_dir(stream) / "*.csv").resolve())
        if suffixes == {".npy"}:
            return "npy", str((_stream_dir(stream) / "*.npy").resolve())
        raise ValueError("SwinMAE files must be all CSV or all NPY.")

    raise ValueError(f"Unsupported stream: {stream}")


class TrainingJob:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state = "idle"
        self._started_at: str | None = None
        self._finished_at: str | None = None
        self._run_id: str | None = None
        self._return_code: int | None = None
        self._message = "Idle"
        self._cmd: list[str] = []
        self._log_tail: list[str] = []
        self._max_log_lines = 400
        self._process: subprocess.Popen[str] | None = None
        self._live_nodes: dict[str, dict[str, str]] = {}
        self._active_step: str | None = None
        self._active_step_index: int | None = None
        self._active_step_total: int | None = None
        self._current_step_nodes: tuple[str, ...] = ()

    def _set_node_status_locked(self, node_id: str, status: str, message: str) -> None:
        self._live_nodes[node_id] = {
            "status": status,
            "message": message,
            "updated_at": utc_now_iso(),
        }

    def _initialize_live_nodes_locked(self) -> None:
        self._live_nodes = {}
        for node_id in DASHBOARD_NODE_IDS:
            self._set_node_status_locked(node_id=node_id, status="idle", message="Pending")

    def _set_current_step_nodes_locked(
        self,
        next_nodes: tuple[str, ...],
        *,
        running_message: str,
        complete_message: str,
    ) -> None:
        previous_nodes = set(self._current_step_nodes)
        next_set = set(next_nodes)

        for node_id in previous_nodes - next_set:
            self._set_node_status_locked(node_id=node_id, status="done", message=complete_message)

        for node_id in next_nodes:
            self._set_node_status_locked(node_id=node_id, status="running", message=running_message)

        self._current_step_nodes = next_nodes

    def _on_step_started_locked(self, *, step_name: str, step_index: int, step_total: int) -> None:
        previous_step = self._active_step
        running_message = f"Step {step_index}/{step_total}: {step_name}"
        complete_message = (
            f"Step complete: {previous_step}" if previous_step else "Stage complete."
        )

        self._active_step = step_name
        self._active_step_index = int(step_index)
        self._active_step_total = int(step_total)
        self._message = running_message
        self._set_node_status_locked(node_id="orchestrator", status="running", message=running_message)
        self._set_current_step_nodes_locked(
            STEP_INITIAL_NODES.get(step_name, ()),
            running_message=running_message,
            complete_message=complete_message,
        )

    def _on_epoch_progress_locked(self, line: str) -> None:
        if not self._active_step:
            return
        trigger = STEP_ACTIVE_NODES_ON_EPOCH.get(self._active_step)
        if trigger is None:
            return

        pattern, promoted_nodes = trigger
        if not pattern.search(line):
            return

        if self._current_step_nodes == promoted_nodes:
            return

        self._set_current_step_nodes_locked(
            promoted_nodes,
            running_message=f"{self._active_step} in progress",
            complete_message="Pre-train data stages complete.",
        )

    def _ingest_progress_line_locked(self, line: str) -> None:
        step_match = PIPELINE_STEP_PATTERN.match(line.strip())
        if step_match:
            self._on_step_started_locked(
                step_name=step_match.group(3),
                step_index=int(step_match.group(1)),
                step_total=int(step_match.group(2)),
            )
            return

        self._on_epoch_progress_locked(line)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "state": self._state,
                "started_at": self._started_at,
                "finished_at": self._finished_at,
                "run_id": self._run_id,
                "return_code": self._return_code,
                "message": self._message,
                "command": self._cmd,
                "log_tail": list(self._log_tail),
                "active_step": self._active_step,
                "step_index": self._active_step_index,
                "step_total": self._active_step_total,
                "live_nodes": {node_id: dict(payload) for node_id, payload in self._live_nodes.items()},
            }

    def start(self, cmd: list[str], run_id: str) -> None:
        with self._lock:
            if self._state in {"running", "stopping"}:
                raise RuntimeError("Training job is already running.")
            self._state = "running"
            self._started_at = utc_now_iso()
            self._finished_at = None
            self._run_id = run_id
            self._return_code = None
            self._message = "Training started."
            self._cmd = list(cmd)
            self._log_tail = []
            self._process = None
            self._active_step = None
            self._active_step_index = None
            self._active_step_total = None
            self._current_step_nodes = ()
            self._initialize_live_nodes_locked()
            self._set_node_status_locked(
                node_id="orchestrator",
                status="running",
                message="Training started.",
            )

    def attach_process(self, process: subprocess.Popen[str]) -> None:
        with self._lock:
            self._process = process

    def append_log(self, line: str) -> None:
        message = line.rstrip("\n")
        if not message:
            return
        with self._lock:
            self._log_tail.append(message)
            if len(self._log_tail) > self._max_log_lines:
                self._log_tail = self._log_tail[-self._max_log_lines :]
            self._message = message
            self._ingest_progress_line_locked(message)

    def finish(self, return_code: int) -> None:
        with self._lock:
            self._state = "success" if return_code == 0 else "failed"
            self._finished_at = utc_now_iso()
            self._return_code = int(return_code)
            if return_code == 0:
                self._message = "Training completed successfully."
                if self._current_step_nodes:
                    self._set_current_step_nodes_locked(
                        (),
                        running_message="",
                        complete_message="Final step complete.",
                    )
                self._set_node_status_locked(
                    node_id="orchestrator",
                    status="done",
                    message="Training completed successfully.",
                )
            else:
                self._message = f"Training failed (rc={return_code})."
                if self._current_step_nodes:
                    for node_id in self._current_step_nodes:
                        self._set_node_status_locked(
                            node_id=node_id,
                            status="fail",
                            message=f"Step failed (rc={return_code}).",
                        )
                self._set_node_status_locked(
                    node_id="orchestrator",
                    status="fail",
                    message=f"Training failed (rc={return_code}).",
                )
            self._process = None

    def fail_fast(self, message: str) -> None:
        with self._lock:
            self._state = "failed"
            self._started_at = self._started_at or utc_now_iso()
            self._finished_at = utc_now_iso()
            self._return_code = -1
            self._message = message
            self._process = None
            self._log_tail.append(message)
            if self._current_step_nodes:
                for node_id in self._current_step_nodes:
                    self._set_node_status_locked(node_id=node_id, status="fail", message=message)
            self._set_node_status_locked(node_id="orchestrator", status="fail", message=message)

    def stop(self) -> bool:
        with self._lock:
            process = self._process
            if self._state not in {"running", "stopping"} or process is None:
                return False
            if process.poll() is not None:
                return False
            self._state = "stopping"
            self._message = "Stop requested. Waiting for process termination..."
            self._set_node_status_locked(
                node_id="orchestrator",
                status="running",
                message="Stop requested. Waiting for process termination...",
            )
            process.terminate()
            return True


TRAINING_JOB = TrainingJob()


def build_training_command(payload: dict[str, Any]) -> tuple[list[str], str]:
    patch_source, patch_path = infer_stream_runtime("patchtst")
    swin_source, swin_path = infer_stream_runtime("swinmae")

    run_id = str(payload.get("run_id") or "").strip()
    if not run_id:
        run_id = datetime.now(timezone.utc).strftime("dashboard-run-%Y%m%d-%H%M%S")

    cmd = [
        sys.executable,
        "-m",
        "pipelines.run_local_training_pipeline",
        "--repo-root",
        str(REPO_ROOT),
        "--patch-config",
        "configs/patchtst_ssl_local.yaml",
        "--swin-config",
        "configs/swinmae_ssl_local.yaml",
        "--patch-data-source",
        patch_source,
        "--patch-data-path",
        patch_path,
        "--swin-data-source",
        swin_source,
        "--swin-data-path",
        swin_path,
        "--run-id",
        run_id,
    ]

    if bool(payload.get("persist_run_history", True)):
        cmd.append("--persist-run-history")
    if bool(payload.get("validate_skip_smoke", True)):
        cmd.append("--validate-skip-smoke")
    if bool(payload.get("run_smoke", False)):
        cmd.append("--run-smoke")

    return cmd, run_id


def start_training_job(payload: dict[str, Any]) -> tuple[bool, str]:
    try:
        cmd, run_id = build_training_command(payload)
    except Exception as exc:
        return False, str(exc)

    try:
        TRAINING_JOB.start(cmd=cmd, run_id=run_id)
    except RuntimeError as exc:
        return False, str(exc)

    def worker() -> None:
        try:
            process = subprocess.Popen(
                cmd,
                cwd=REPO_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            TRAINING_JOB.attach_process(process)

            if process.stdout is not None:
                for line in process.stdout:
                    TRAINING_JOB.append_log(line)

            rc = process.wait()
            TRAINING_JOB.finish(return_code=rc)
        except Exception as exc:
            TRAINING_JOB.fail_fast(f"Training crashed: {exc}")

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    return True, "Training started"


class DashboardRequestHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, directory=str(DASHBOARD_ROOT), **kwargs)

    def _send_json(self, status_code: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json_body(self) -> dict[str, Any]:
        content_length = int(self.headers.get("Content-Length", "0") or 0)
        if content_length <= 0:
            return {}
        raw = self.rfile.read(content_length)
        if not raw:
            return {}
        payload = json.loads(raw.decode("utf-8"))
        if isinstance(payload, dict):
            return payload
        return {}

    def _status_payload(self) -> dict[str, Any]:
        return {
            "job": TRAINING_JOB.snapshot(),
            "uploads": summarize_uploads(),
            "repo_root": str(REPO_ROOT),
        }

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/status":
            self._send_json(HTTPStatus.OK, self._status_payload())
            return

        if parsed.path == "/api/uploads":
            self._send_json(HTTPStatus.OK, {"uploads": summarize_uploads()})
            return

        super().do_GET()

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/upload":
            self._handle_upload(parsed)
            return

        if parsed.path == "/api/train":
            self._handle_train()
            return

        if parsed.path == "/api/stop":
            self._handle_stop()
            return

        self._send_json(HTTPStatus.NOT_FOUND, {"error": "Unknown API endpoint."})

    def _handle_upload(self, parsed) -> None:
        query = parse_qs(parsed.query)
        stream = (query.get("stream") or [""])[0].strip().lower()
        replace = (query.get("replace") or ["1"])[0] != "0"

        if stream not in STREAM_SPECS:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "stream must be one of: patchtst, swinmae"})
            return

        content_type = self.headers.get("Content-Type", "")
        if "multipart/form-data" not in content_type:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Content-Type must be multipart/form-data"})
            return

        if replace:
            stream_dir = _stream_dir(stream)
            if stream_dir.exists():
                shutil.rmtree(stream_dir)
            stream_dir.mkdir(parents=True, exist_ok=True)

        form = cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={
                "REQUEST_METHOD": "POST",
                "CONTENT_TYPE": content_type,
            },
        )

        files_field = form["files"] if "files" in form else None
        if files_field is None:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "No files provided. Use field name 'files'."})
            return

        file_items = files_field if isinstance(files_field, list) else [files_field]
        saved: list[str] = []
        try:
            for item in file_items:
                filename = getattr(item, "filename", "") or ""
                if not filename:
                    continue
                blob = item.file.read()
                if not blob:
                    continue
                saved_paths = save_uploaded_blob(stream=stream, filename=filename, blob=blob)
                saved.extend(str(path.relative_to(REPO_ROOT)) for path in saved_paths)
        except Exception as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            return

        if not saved:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "No valid files were uploaded."})
            return

        payload = self._status_payload()
        payload["saved_files"] = saved
        self._send_json(HTTPStatus.OK, payload)

    def _handle_train(self) -> None:
        try:
            payload = self._read_json_body()
        except Exception as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": f"Invalid JSON body: {exc}"})
            return

        started, message = start_training_job(payload)
        if not started:
            status = HTTPStatus.CONFLICT if "already running" in message.lower() else HTTPStatus.BAD_REQUEST
            self._send_json(status, {"error": message, **self._status_payload()})
            return

        status_payload = self._status_payload()
        status_payload["message"] = message
        self._send_json(HTTPStatus.ACCEPTED, status_payload)

    def _handle_stop(self) -> None:
        stopped = TRAINING_JOB.stop()
        if not stopped:
            self._send_json(HTTPStatus.CONFLICT, {"error": "No running job to stop.", **self._status_payload()})
            return
        status_payload = self._status_payload()
        status_payload["message"] = "Stop requested."
        self._send_json(HTTPStatus.OK, status_payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve dashboard UI + training control API.")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for spec in STREAM_SPECS.values():
        Path(spec["dir"]).mkdir(parents=True, exist_ok=True)

    server = ThreadingHTTPServer((args.host, int(args.port)), DashboardRequestHandler)
    print(f"Dashboard server running at http://{args.host}:{args.port}")
    print("Use /api/status to inspect training status and uploads.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
