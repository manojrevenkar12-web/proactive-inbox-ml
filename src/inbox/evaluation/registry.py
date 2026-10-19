"""A file-based model registry: every candidate with its metrics, data hash, gate decision and status.

Deliberately simple (one JSON file) so it runs anywhere; the same record maps one-to-one onto an
MLflow registry entry (params, metrics, tags, alias "champion")."""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path


def _git_sha() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True).stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


class Registry:
    def __init__(self, path: Path):
        self.path = path
        self.entries: list[dict] = json.loads(path.read_text()) if path.exists() else []

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.entries, indent=2))

    def champion(self, task: str) -> dict | None:
        return next((e for e in reversed(self.entries) if e["task"] == task and e["status"] == "champion"), None)

    def add(self, task: str, name: str, artifact: str | None, metrics: dict, data_sha: str, decision: str, reasons: list[str]) -> dict:
        if decision == "champion":
            for e in self.entries:
                if e["task"] == task and e["status"] == "champion":
                    e["status"] = "retired"
        entry = {
            "task": task,
            "name": name,
            "version": f"{task}-{len([e for e in self.entries if e['task'] == task]) + 1}",
            "artifact": artifact,
            "metrics": metrics,
            "data_sha256": data_sha,
            "git_sha": _git_sha(),
            "created": datetime.now(UTC).isoformat(timespec="seconds"),
            "status": decision,
            "gate_reasons": reasons,
        }
        self.entries.append(entry)
        self.save()
        return entry
