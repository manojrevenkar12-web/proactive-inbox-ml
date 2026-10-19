import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
# Share the downloaded dataset across test runs; everything else goes to a temp dir.
os.environ.setdefault("INBOX_DATA", str(ROOT / "artifacts" / "data"))


@pytest.fixture(scope="session")
def trained(tmp_path_factory) -> Path:
    """A fast end-to-end training run (about a minute on CPU), shared by the API tests."""
    home = tmp_path_factory.mktemp("inbox")
    os.environ["INBOX_HOME"] = str(home)
    from inbox import pipeline

    report = pipeline.run(fast=True, log=lambda *_: None)
    (home / "report.json").write_text(__import__("json").dumps(report, default=str))
    return home
