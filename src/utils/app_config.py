"""Load shared settings from config/config.yaml (overridable via environment)."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Dict

import yaml

from src.utils.project_paths import repo_root


@lru_cache(maxsize=1)
def _yaml_config() -> Dict[str, Any]:
    path = repo_root() / "config" / "config.yaml"
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError("config/config.yaml must parse to a mapping")
    return data


def student_model_id() -> str:
    """Hugging Face id for the student; override with env STUDENT_MODEL if set."""
    override = os.getenv("STUDENT_MODEL", "").strip()
    if override:
        return override
    return str(_yaml_config()["models"]["student_model"])
