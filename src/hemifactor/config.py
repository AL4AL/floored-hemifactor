from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "default.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    with cfg_path.open() as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {cfg_path}")
    data["_config_path"] = str(cfg_path)
    data["_root"] = str(ROOT)
    return data


def artifacts_dir(cfg: dict[str, Any]) -> Path:
    root = Path(cfg["_root"])
    rel = cfg.get("paths", {}).get("artifacts_dir", "artifacts")
    out = root / rel
    out.mkdir(parents=True, exist_ok=True)
    return out
