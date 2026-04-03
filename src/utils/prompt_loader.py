"""Load prompt templates from repo-root files (Assignment §6: editable templates, not only inline)."""

from pathlib import Path
from typing import Dict

_REPO_ROOT = Path(__file__).resolve().parents[2]


def repo_root() -> Path:
    return _REPO_ROOT


def load_prompt(relative_path: str) -> str:
    p = _REPO_ROOT / relative_path
    if not p.is_file():
        raise FileNotFoundError(f"Prompt template not found: {p}")
    return p.read_text(encoding="utf-8").strip()


def fill_placeholders(template: str, mapping: Dict[str, str]) -> str:
    out = template
    for key, value in mapping.items():
        out = out.replace(f"__{key.upper()}__", value)
    return out
