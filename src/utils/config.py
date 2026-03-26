from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

import yaml


def _deep_update(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    merged = deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(config_path: str | Path, base_path: str | Path | None = None) -> Dict[str, Any]:
    config_path = Path(config_path)
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    if base_path is not None:
        with Path(base_path).open("r", encoding="utf-8") as f:
            base_cfg = yaml.safe_load(f) or {}
        return _deep_update(base_cfg, cfg)
    return cfg

