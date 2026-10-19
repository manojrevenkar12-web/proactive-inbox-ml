"""Serving-side classifier: ONNX Runtime + numpy only (no PyTorch in the inference image)."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

from .. import features
from ..data.clinc import action
from .calibrate import Decision, decide, softmax


class ActionClassifier:
    def __init__(self, model_path: Path, meta_path: Path, threads: int = 1):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(model_path), opts, providers=["CPUExecutionProvider"])
        meta = json.loads(meta_path.read_text())
        self.labels: list[str] = meta["labels"]
        self.version: str = meta["version"]
        self.char_ngrams: bool = meta.get("char_ngrams", True)
        self.arch: str = meta.get("arch", "ngram")
        self.decision = Decision(meta["temperature"], meta["threshold"], self.labels.index("oos") if "oos" in self.labels else None)

    def logits(self, texts: list[str]) -> np.ndarray:
        ids = features.sequence(texts) if self.arch == "transformer" else features.padded(texts, self.char_ngrams)
        return self.session.run(["logits"], {"ids": ids})[0]

    def predict(self, texts: list[str]) -> list[dict]:
        probs = softmax(self.logits(texts), self.decision.temperature)
        idx = decide(probs, self.decision)
        out = []
        for row, i in zip(probs, idx, strict=True):
            if i < 0:
                out.append({"intent": "oos", "action": "none", "confidence": round(float(1 - row.max()), 4), "out_of_scope": True})
            else:
                out.append({"intent": self.labels[i], "action": action(self.labels[i]), "confidence": round(float(row[i]), 4), "out_of_scope": False})
        return out


def latency(predict, texts: list[str], batch: int = 1, repeats: int = 1) -> dict:
    """p50/p95/p99 in milliseconds per call, on the calling thread."""
    times = []
    for _ in range(repeats):
        for s in range(0, len(texts), batch):
            t0 = time.perf_counter()
            predict(texts[s : s + batch])
            times.append((time.perf_counter() - t0) * 1000)
    a = np.array(times)
    return {
        "batch": batch,
        "calls": len(a),
        "p50_ms": round(float(np.percentile(a, 50)), 3),
        "p95_ms": round(float(np.percentile(a, 95)), 3),
        "p99_ms": round(float(np.percentile(a, 99)), 3),
        "per_item_ms": round(float(a.mean() / batch), 4),
    }
