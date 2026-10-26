from __future__ import annotations

import json
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_config(path: str | Path | None = None) -> dict:
    root = project_root()
    cfg_path = Path(path) if path is not None else root / "config" / "experiment.json"
    if not cfg_path.is_absolute():
        cfg_path = root / cfg_path
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["_project_root"] = str(root)
    return cfg


def resolve_path(cfg: dict, key: str) -> Path:
    root = Path(cfg["_project_root"])
    p = Path(cfg["paths"][key])
    return p if p.is_absolute() else root / p
