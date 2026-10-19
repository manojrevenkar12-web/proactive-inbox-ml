"""CLINC150 (Larson et al., EMNLP 2019): 150 intents across 10 domains plus out-of-scope
queries, written by crowd workers. CC BY 3.0. Downloaded from a pinned commit and verified by
SHA-256, so every run trains and evaluates on exactly the same data.

Out-of-scope matters for an assistant that acts proactively: acting on a message it does not
understand is worse than doing nothing.
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from ..config import data_dir

COMMIT = "828f8093932c8fe6ca7936c3d2e52903b1c523de"
FILES = {
    "data_full.json": "36923c3705a59e08fe9c3883d8bc2dd966ef93e22cb78ac41171782a698d56e0",
    "domains.json": None,  # verified through the intents it must cover
}
URL = "https://raw.githubusercontent.com/clinc/oos-eval/{commit}/data/{name}"
OOS = "oos"

# What the messaging app does with an intent. Everything else is answered, not acted on.
ACTIONS = {
    "reminder": "set_reminder",
    "reminder_update": "set_reminder",
    "calendar": "check_calendar",
    "calendar_update": "add_event",
    "meeting_schedule": "add_event",
    "schedule_meeting": "add_event",
    "todo_list": "check_tasks",
    "todo_list_update": "add_task",
    "shopping_list": "check_tasks",
    "shopping_list_update": "add_task",
    "alarm": "set_alarm",
    "timer": "set_alarm",
    "text": "send_message",
    "make_call": "start_call",
    "share_location": "share_location",
    "pay_bill": "pay_bill",
    "transfer": "pay_bill",
    "translate": "translate",
    "definition": "answer",
}


@dataclass(frozen=True)
class Split:
    texts: list[str]
    labels: list[str]


def _download(name: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(URL.format(commit=COMMIT, name=name), timeout=60) as r:  # noqa: S310 - pinned https URL
        dest.write_bytes(r.read())


def ensure(directory: Path | None = None) -> Path:
    d = directory or data_dir() / "clinc150"
    for name, sha in FILES.items():
        path = d / name
        if not path.exists():
            _download(name, path)
        if sha and hashlib.sha256(path.read_bytes()).hexdigest() != sha:
            raise ValueError(f"{path} does not match the pinned SHA-256; delete it and fetch again")
    return d


def load(directory: Path | None = None) -> dict[str, Split]:
    d = ensure(directory)
    raw = json.loads((d / "data_full.json").read_text())

    def split(name: str) -> Split:
        rows = raw[name] + raw[f"oos_{name}"]
        return Split([t for t, _ in rows], [lab for _, lab in rows])

    return {"train": split("train"), "val": split("val"), "test": split("test")}


def domains(directory: Path | None = None) -> dict[str, str]:
    """intent -> domain (banking, travel, kitchen_and_dining, …); oos -> oos."""
    d = ensure(directory)
    raw = json.loads((d / "domains.json").read_text())
    out = {intent: dom for dom, intents in raw.items() for intent in intents}
    out[OOS] = OOS
    return out


def action(intent: str) -> str:
    if intent == OOS:
        return "none"
    return ACTIONS.get(intent, "answer")
