"""Classification metrics for intent + out-of-scope, with bootstrap confidence intervals and slices.

Protocol follows the CLINC150 paper: in-scope accuracy counts a prediction of "out-of-scope" on
an in-scope query as an error; out-of-scope recall is the share of out-of-scope queries the
system declines to act on.
"""

from __future__ import annotations

import numpy as np

OOS = "oos"


def per_item(pred: list[str], gold: list[str]) -> dict[str, np.ndarray]:
    p, g = np.array(pred), np.array(gold)
    ins = g != OOS
    return {"inscope_correct": (p == g)[ins], "oos_detected": (p == OOS)[~ins], "false_oos": (p == OOS)[ins]}


def macro_f1(pred: list[str], gold: list[str]) -> float:
    labels = sorted(set(gold))
    p, g = np.array(pred), np.array(gold)
    f1s = []
    for lab in labels:
        tp = np.sum((p == lab) & (g == lab))
        fp = np.sum((p == lab) & (g != lab))
        fn = np.sum((p != lab) & (g == lab))
        f1s.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
    return float(np.mean(f1s))


def bootstrap_ci(x: np.ndarray, n: int = 1000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n, len(x)), replace=True).mean(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def report(pred: list[str], gold: list[str], domain_of: dict[str, str] | None = None) -> dict:
    it = per_item(pred, gold)
    p, g = np.array(pred), np.array(gold)
    oos_pred = p == OOS
    out = {
        "inscope_accuracy": round(float(it["inscope_correct"].mean()), 4),
        "inscope_accuracy_ci95": [round(v, 4) for v in bootstrap_ci(it["inscope_correct"].astype(float))],
        "oos_recall": round(float(it["oos_detected"].mean()), 4),
        "oos_recall_ci95": [round(v, 4) for v in bootstrap_ci(it["oos_detected"].astype(float))],
        "oos_precision": round(float((g[oos_pred] == OOS).mean()), 4) if oos_pred.any() else None,
        "false_oos_rate": round(float(it["false_oos"].mean()), 4),
        "macro_f1": round(macro_f1(pred, gold), 4),
        "n_inscope": int((g != OOS).sum()),
        "n_oos": int((g == OOS).sum()),
    }
    if domain_of:
        slices = {}
        for dom in sorted(set(domain_of.values()) - {OOS}):
            m = np.array([domain_of.get(x) == dom for x in gold])
            slices[dom] = round(float((p[m] == g[m]).mean()), 4)
        out["domain_accuracy"] = slices
    return out
