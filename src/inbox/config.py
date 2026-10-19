"""Paths. Everything the pipeline writes goes under INBOX_HOME (default: ./artifacts)."""

from __future__ import annotations

import os
from pathlib import Path


def home() -> Path:
    return Path(os.environ.get("INBOX_HOME", "artifacts"))


def data_dir() -> Path:
    return Path(os.environ.get("INBOX_DATA", home() / "data"))


def models_dir() -> Path:
    return home() / "models"


def reports_dir() -> Path:
    return home() / "reports"
