"""Command line: `inbox data | train | serve | loadtest`."""

from __future__ import annotations

import argparse
import json
import sys


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="inbox")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("data", help="download and verify CLINC150 (pinned commit + SHA-256)")
    t = sub.add_parser("train", help="train, evaluate, gate and register every model")
    t.add_argument("--fast", action="store_true", help="small, quick run (CI)")
    t.add_argument("--with-transformer", action="store_true", help="also train the transformer challenger (slow on CPU)")
    e = sub.add_parser("dp-scale", help="experiment: privacy cost of federated personalisation vs population size")
    e.add_argument("--populations", default="2000,10000")
    s = sub.add_parser("serve", help="inference API")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    lt = sub.add_parser("loadtest", help="concurrent load against a running API")
    lt.add_argument("--url", default="http://127.0.0.1:8000")
    lt.add_argument("--requests", type=int, default=2000)
    lt.add_argument("--concurrency", type=int, default=16)
    a = p.parse_args(argv)

    if a.cmd == "data":
        from .data import clinc

        print(clinc.ensure())
    elif a.cmd == "train":
        from . import pipeline

        pipeline.run(fast=a.fast, with_transformer=a.with_transformer)
    elif a.cmd == "dp-scale":
        from . import pipeline

        pipeline.dp_scale(tuple(int(x) for x in a.populations.split(",")))
    elif a.cmd == "serve":
        import uvicorn

        uvicorn.run("inbox.serving.app:create_app", factory=True, host=a.host, port=a.port, log_level="warning")
    elif a.cmd == "loadtest":
        from .serving.loadtest import run

        print(json.dumps(run(a.url, a.requests, a.concurrency), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
