"""Load prompt templates from repo-root files (editable templates, not only inline strings)."""

from pathlib import Path
from typing import Dict

from src.utils.project_paths import repo_root


def load_prompt(relative_path: str) -> str:
    p: Path = repo_root() / relative_path
    if not p.is_file():
        raise FileNotFoundError(f"Prompt template not found: {p}")
    return p.read_text(encoding="utf-8").strip()


def fill_placeholders(template: str, mapping: Dict[str, str]) -> str:
    out = template
    for key, value in mapping.items():
        out = out.replace(f"__{key.upper()}__", value)
    return out
