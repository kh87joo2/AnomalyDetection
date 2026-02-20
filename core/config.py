from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    pass


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ConfigError(f"Config root must be a mapping: {path}")
    return data


def get_required(config: dict[str, Any], dotted_key: str) -> Any:
    current: Any = config
    for key in dotted_key.split("."):
        if not isinstance(current, dict) or key not in current:
            raise ConfigError(f"Missing required config key: {dotted_key}")
        current = current[key]
    return current


def validate_required_keys(config: dict[str, Any], required_keys: list[str]) -> None:
    for key in required_keys:
        get_required(config, key)
