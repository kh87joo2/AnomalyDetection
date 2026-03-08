from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import torch

from batch_decision.contracts import (
    ArtifactPaths,
    BatchRunRequest,
    BatchScorePayload,
    PreparedBatch,
    StreamName,
    StreamScorePayload,
    WindowScore,
)
from batch_decision.preprocess import prepare_patchtst_batch, prepare_swinmae_batch
from core.config import ConfigError, load_yaml_config
from datasets.transforms.cwt import vibration_window_to_image
from datasets.transforms.fdc_normalization import ChannelScaler
from inference.checkpoint_io import load_checkpoint
from inference.scoring import infer_score
from models.patchtst.patchtst_ssl import PatchTSTSSL
from models.swinmae.swinmae_ssl import SwinMAESSL


class BatchScoringError(ValueError):
    pass


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


def _select_device(config: dict[str, Any]) -> torch.device:
    prefer_cuda = bool(config.get("device", {}).get("prefer_cuda", True))
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _build_model(
    stream: Literal["patchtst", "swinmae"],
    config: dict[str, Any],
    device: torch.device,
) -> torch.nn.Module:
    model_cfg = config["model"]
    if stream == "patchtst":
        data_cfg = config["data"]
        model = PatchTSTSSL(
            seq_len=int(data_cfg["seq_len"]),
            patch_len=int(model_cfg["patch_len"]),
            patch_stride=int(model_cfg["patch_stride"]),
            d_model=int(model_cfg["d_model"]),
            nhead=int(model_cfg["nhead"]),
            num_layers=int(model_cfg["num_layers"]),
            ff_dim=int(model_cfg["ff_dim"]),
            dropout=float(model_cfg["dropout"]),
            mask_ratio=float(model_cfg["mask_ratio"]),
        )
    else:
        model = SwinMAESSL(
            mask_ratio=float(model_cfg["mask_ratio"]),
            patch_size=int(model_cfg["patch_size"]),
            use_timm_swin=bool(model_cfg.get("use_timm_swin", True)),
            timm_name=str(model_cfg.get("timm_name", "swin_tiny_patch4_window7_224")),
            decoder_dim=int(model_cfg.get("decoder_dim", 256)),
        )
    return model.to(device)


def _extract_model_state_dict(checkpoint: Any) -> dict[str, Any]:
    if not isinstance(checkpoint, dict):
        raise BatchScoringError("Checkpoint must be a mapping.")
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    if not isinstance(state_dict, dict):
        raise BatchScoringError("Checkpoint does not contain a valid model_state_dict.")
    return state_dict


def _load_model(
    *,
    stream: Literal["patchtst", "swinmae"],
    config: dict[str, Any],
    checkpoint_path: str,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    checkpoint_file = Path(checkpoint_path)
    if not checkpoint_file.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_file}")

    model = _build_model(stream, config, device)
    checkpoint = load_checkpoint(checkpoint_file, map_location=device)
    state_dict = _extract_model_state_dict(checkpoint)
    incompatible = model.load_state_dict(state_dict, strict=False)
    model.eval()

    return model, {
        "checkpoint_path": str(checkpoint_file.resolve()),
        "missing_keys": len(incompatible.missing_keys),
        "unexpected_keys": len(incompatible.unexpected_keys),
        "device": str(device),
    }


def _to_python(value: Any) -> Any:
    if torch.is_tensor(value):
        tensor = value.detach().cpu()
        if tensor.ndim == 0:
            return float(tensor.item())
        return tensor.tolist()
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return float(value.item())
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _to_python(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_python(v) for v in value]
    return value


def _slice_aux(value: Any, idx: int, batch_size: int) -> Any:
    if torch.is_tensor(value):
        tensor = value.detach().cpu()
        if tensor.ndim > 0 and tensor.shape[0] == batch_size:
            return _to_python(tensor[idx])
        return _to_python(tensor)
    if isinstance(value, np.ndarray):
        if value.ndim > 0 and value.shape[0] == batch_size:
            return _to_python(value[idx])
        return _to_python(value)
    return _to_python(value)


def _records_from_chunk(
    *,
    prepared: PreparedBatch,
    stream: Literal["patchtst", "swinmae"],
    offset: int,
    score: torch.Tensor,
    aux: dict[str, Any],
) -> list[WindowScore]:
    batch_size = int(score.shape[0])
    records: list[WindowScore] = []
    for local_idx in range(batch_size):
        window_index = offset + local_idx
        aux_record = {
            key: _slice_aux(value, local_idx, batch_size)
            for key, value in aux.items()
        }
        records.append(
            WindowScore(
                event_id=f"{stream}:{window_index:06d}",
                stream=stream,
                file_id=prepared.window_file_ids[window_index],
                timestamp=prepared.window_anchor_timestamps[window_index],
                window_index=window_index,
                score=float(score[local_idx].detach().cpu().item()),
                aux=aux_record,
            )
        )
    return records


def _score_patchtst_windows(
    prepared: PreparedBatch,
    *,
    config: dict[str, Any],
    artifacts: ArtifactPaths,
) -> StreamScorePayload:
    if not artifacts.patchtst_checkpoint:
        raise BatchScoringError("PatchTST scoring requires artifact_paths.patchtst_checkpoint")
    if not artifacts.scaler_path:
        raise BatchScoringError("PatchTST scoring requires artifact_paths.scaler")

    scaler_file = Path(artifacts.scaler_path)
    if not scaler_file.exists():
        raise FileNotFoundError(f"Scaler artifact not found: {scaler_file}")

    device = _select_device(config)
    model, load_info = _load_model(
        stream="patchtst",
        config=config,
        checkpoint_path=artifacts.patchtst_checkpoint,
        device=device,
    )
    scaler = ChannelScaler.load(scaler_file)
    normalized = scaler.transform(prepared.windows)

    batch_size = int(config.get("training", {}).get("batch_size", 16))
    records: list[WindowScore] = []
    for start in range(0, normalized.shape[0], batch_size):
        chunk = torch.from_numpy(normalized[start : start + batch_size]).to(device)
        output = infer_score(batch=chunk, model=model, stream="patchtst")
        records.extend(
            _records_from_chunk(
                prepared=prepared,
                stream="patchtst",
                offset=start,
                score=output["score"],
                aux=output["aux"],
            )
        )

    return StreamScorePayload(
        stream="patchtst",
        records=records,
        metadata={
            "window_count": len(records),
            "scaler_path": str(scaler_file.resolve()),
            "normalization": scaler.method,
            **load_info,
        },
    )


def _window_to_image(window: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    return vibration_window_to_image(
        window=window,
        fs=int(config["data"]["fs"]),
        freq_min=float(config["cwt"]["freq_min"]),
        freq_max=float(config["cwt"]["freq_max"]),
        n_freqs=int(config["cwt"]["n_freqs"]),
        image_size=int(config["image"]["size"]),
        wavelet=str(config["cwt"].get("wavelet", "morl")),
        log_mag=bool(config["cwt"].get("log_mag", True)),
        normalize=str(config["cwt"].get("normalize", "robust")),
    )


def _score_swinmae_windows(
    prepared: PreparedBatch,
    *,
    config: dict[str, Any],
    artifacts: ArtifactPaths,
) -> StreamScorePayload:
    if not artifacts.swinmae_checkpoint:
        raise BatchScoringError("SwinMAE scoring requires artifact_paths.swinmae_checkpoint")

    device = _select_device(config)
    model, load_info = _load_model(
        stream="swinmae",
        config=config,
        checkpoint_path=artifacts.swinmae_checkpoint,
        device=device,
    )

    batch_size = int(config.get("training", {}).get("batch_size", 8))
    records: list[WindowScore] = []
    for start in range(0, prepared.windows.shape[0], batch_size):
        chunk_windows = prepared.windows[start : start + batch_size]
        chunk_images = np.stack(
            [_window_to_image(window, config) for window in chunk_windows],
            axis=0,
        ).astype(np.float32)
        batch = torch.from_numpy(chunk_images).to(device)
        output = infer_score(batch=batch, model=model, stream="swinmae")
        records.extend(
            _records_from_chunk(
                prepared=prepared,
                stream="swinmae",
                offset=start,
                score=output["score"],
                aux=output["aux"],
            )
        )

    return StreamScorePayload(
        stream="swinmae",
        records=records,
        metadata={
            "window_count": len(records),
            "cwt": copy.deepcopy(config["cwt"]),
            "image": copy.deepcopy(config["image"]),
            **load_info,
        },
    )


def score_windows(
    prepared: PreparedBatch,
    *,
    stream: Literal["patchtst", "swinmae"],
    config: dict[str, Any],
    artifacts: ArtifactPaths,
) -> StreamScorePayload:
    if prepared.stream != stream:
        raise BatchScoringError(
            f"Prepared batch stream mismatch: prepared={prepared.stream} requested={stream}"
        )
    if stream == "patchtst":
        return _score_patchtst_windows(prepared, config=config, artifacts=artifacts)
    return _score_swinmae_windows(prepared, config=config, artifacts=artifacts)


def _load_stream_config(
    *,
    runtime_config: dict[str, Any],
    runtime_config_path: Path | None,
    input_path: str,
    stream: Literal["patchtst", "swinmae"],
) -> dict[str, Any]:
    preprocess_cfg = runtime_config.get("preprocess")
    if not isinstance(preprocess_cfg, dict):
        raise ConfigError("Runtime config must include preprocess mapping for score-only execution")

    key = "patchtst_config" if stream == "patchtst" else "swinmae_config"
    raw_cfg_path = preprocess_cfg.get(key)
    if not isinstance(raw_cfg_path, str) or not raw_cfg_path.strip():
        raise ConfigError(f"Runtime config must include preprocess.{key}")

    cfg_path = _resolve_path(raw_cfg_path.strip(), runtime_config_path)
    config = load_yaml_config(cfg_path)
    data_cfg = config.setdefault("data", {})
    data_cfg["path"] = input_path
    return config


def score_batch_request(
    request: BatchRunRequest,
    *,
    runtime_config: dict[str, Any],
    runtime_config_path: Path | None = None,
) -> BatchScorePayload:
    patchtst_records: list[WindowScore] = []
    swinmae_records: list[WindowScore] = []
    metadata: dict[str, Any] = {}

    if request.stream in {"patchtst", "dual"}:
        if not request.input_paths.patchtst:
            raise BatchScoringError("Missing patchtst input path for scoring")
        patchtst_cfg = _load_stream_config(
            runtime_config=runtime_config,
            runtime_config_path=runtime_config_path,
            input_path=request.input_paths.patchtst,
            stream="patchtst",
        )
        patchtst_batch = prepare_patchtst_batch(patchtst_cfg)
        patchtst_scores = score_windows(
            patchtst_batch,
            stream="patchtst",
            config=patchtst_cfg,
            artifacts=request.artifacts,
        )
        patchtst_records = patchtst_scores.records
        metadata["patchtst"] = {
            "scored_windows": len(patchtst_records),
            "skipped_files": list(patchtst_batch.skipped_files),
            "dqvl_reports": len(patchtst_batch.dqvl_records),
            **patchtst_scores.metadata,
        }

    if request.stream in {"swinmae", "dual"}:
        if not request.input_paths.swinmae:
            raise BatchScoringError("Missing swinmae input path for scoring")
        swinmae_cfg = _load_stream_config(
            runtime_config=runtime_config,
            runtime_config_path=runtime_config_path,
            input_path=request.input_paths.swinmae,
            stream="swinmae",
        )
        swinmae_batch = prepare_swinmae_batch(swinmae_cfg)
        swinmae_scores = score_windows(
            swinmae_batch,
            stream="swinmae",
            config=swinmae_cfg,
            artifacts=request.artifacts,
        )
        swinmae_records = swinmae_scores.records
        metadata["swinmae"] = {
            "scored_windows": len(swinmae_records),
            "skipped_files": list(swinmae_batch.skipped_files),
            "dqvl_reports": len(swinmae_batch.dqvl_records),
            **swinmae_scores.metadata,
        }

    return BatchScorePayload(
        run_id=request.run_id,
        stream=cast(StreamName, request.stream),
        patchtst_records=patchtst_records,
        swinmae_records=swinmae_records,
        metadata=metadata,
    )
