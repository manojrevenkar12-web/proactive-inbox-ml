"""Temperature scaling and the out-of-scope threshold.

The classifier acts only when it is confident. Confidence is the calibrated probability of the
best in-scope intent; below the threshold the message is treated as out-of-scope and the app
does nothing. The threshold is chosen on the validation split, never on test, with an explicit
budget for how much in-scope accuracy we are willing to give up to catch out-of-scope messages.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def softmax(z: np.ndarray, t: float = 1.0) -> np.ndarray:
    z = z / t
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    """Minimise validation negative log-likelihood over T (1-D search, robust and dependency-free)."""
    best_t, best_nll = 1.0, np.inf
    for t in np.exp(np.linspace(np.log(0.05), np.log(10), 200)):
        p = softmax(logits, t)[np.arange(len(y)), y]
        nll = -np.log(np.clip(p, 1e-12, None)).mean()
        if nll < best_nll:
            best_t, best_nll = float(t), nll
    return best_t


def ece(probs: np.ndarray, y: np.ndarray, bins: int = 15) -> float:
    conf, pred = probs.max(1), probs.argmax(1)
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(total)


@dataclass(frozen=True)
class Decision:
    temperature: float
    threshold: float
    oos_index: int | None


def decide(probs: np.ndarray, d: Decision) -> np.ndarray:
    """Predicted class index per row; -1 means out-of-scope (do nothing)."""
    p = probs.copy()
    if d.oos_index is not None:
        oos_p = p[:, d.oos_index].copy()
        p[:, d.oos_index] = -1
    else:
        oos_p = np.zeros(len(p))
    best = p.argmax(1)
    conf = p.max(1)
    out = np.where((conf < d.threshold) | (oos_p > conf), -1, best)
    return out


def choose_threshold(
    probs: np.ndarray, y: np.ndarray, oos_index: int | None, temperature: float, max_inscope_drop: float = 0.02
) -> tuple[Decision, list[dict]]:
    """Largest out-of-scope recall whose in-scope accuracy is within `max_inscope_drop` of the
    no-threshold accuracy. y uses -1 for out-of-scope rows."""
    ins, oos = y >= 0, y < 0
    curve = []
    base_acc = None
    for th in np.round(np.arange(0.0, 0.96, 0.01), 2):
        pred = decide(probs, Decision(temperature, float(th), oos_index))
        acc = float((pred[ins] == y[ins]).mean())
        rec = float((pred[oos] == -1).mean()) if oos.any() else 0.0
        base_acc = acc if base_acc is None else base_acc
        curve.append({"threshold": float(th), "inscope_accuracy": round(acc, 4), "oos_recall": round(rec, 4)})
    ok = [c for c in curve if c["inscope_accuracy"] >= curve[0]["inscope_accuracy"] - max_inscope_drop]
    best = max(ok, key=lambda c: (c["oos_recall"], c["inscope_accuracy"]))
    return Decision(temperature, best["threshold"], oos_index), curve
