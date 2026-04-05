"""Repository root and sys.path bootstrap for script-style entrypoints."""

from __future__ import annotations

import sys
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def ensure_repo_on_path() -> Path:
    root = repo_root()
    r = str(root)
    if r not in sys.path:
        sys.path.insert(0, r)
    return root
