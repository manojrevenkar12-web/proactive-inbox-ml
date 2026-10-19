"""Release gate: a challenger replaces the champion only if it is not worse beyond noise.

Paired bootstrap over the same test items: for each key metric, the 95% interval of
(challenger − champion) must not fall entirely below −tolerance, and no slice (domain) may drop
by more than `slice_tolerance`. Latency and model size have hard budgets. The gate's decision and
every reason are stored with the model in the registry, so a rejected model is explained.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class GateResult:
    promote: bool
    reasons: list[str] = field(default_factory=list)
    deltas: dict = field(default_factory=dict)


def paired_delta(champ: np.ndarray, chall: np.ndarray, n: int = 2000, seed: int = 1) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(champ), size=(n, len(champ)))
    d = chall[idx].mean(1) - champ[idx].mean(1)
    return float(chall.mean() - champ.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def compare(
    champion: dict,
    challenger: dict,
    tolerance: float = 0.005,
    slice_tolerance: float = 0.03,
    latency_budget_ms: float = 5.0,
    size_budget_mb: float = 50.0,
) -> GateResult:
    """champion/challenger: {"items": {metric: per-item 0/1}, "slice_items": {slice: per-item 0/1}, "latency_p95_ms": float, "size_mb": float}"""
    res = GateResult(promote=True)
    better = False
    for metric, arr in challenger["items"].items():
        mean, lo, hi = paired_delta(champion["items"][metric].astype(float), arr.astype(float))
        res.deltas[metric] = {"delta": round(mean, 4), "ci95": [round(lo, 4), round(hi, 4)]}
        if hi < -tolerance:
            res.promote = False
            res.reasons.append(f"{metric} is worse: Δ {mean:+.4f} (95% CI {lo:+.4f} … {hi:+.4f})")
        if lo > 0:
            better = True
    for dom, arr in challenger.get("slice_items", {}).items():
        before = champion.get("slice_items", {}).get(dom)
        if before is None:
            continue
        mean, lo, hi = paired_delta(before.astype(float), arr.astype(float))
        # a slice blocks the release only if it is significantly worse AND the drop is material
        if hi < 0 and mean < -slice_tolerance:
            res.promote = False
            res.reasons.append(f"slice {dom} is worse: Δ {mean:+.3f} (95% CI {lo:+.3f} … {hi:+.3f})")
    if challenger["latency_p95_ms"] > latency_budget_ms:
        res.promote = False
        res.reasons.append(f"p95 latency {challenger['latency_p95_ms']:.2f} ms over the {latency_budget_ms} ms budget")
    if challenger["size_mb"] > size_budget_mb:
        res.promote = False
        res.reasons.append(f"model size {challenger['size_mb']:.1f} MB over the {size_budget_mb} MB budget")
    if res.promote and not better:
        cheaper = challenger["latency_p95_ms"] < champion["latency_p95_ms"] * 0.8 or challenger["size_mb"] < champion["size_mb"] * 0.8
        if not cheaper:
            res.promote = False
            res.reasons.append("no metric significantly better and not meaningfully cheaper: keep the champion")
        else:
            res.reasons.append("quality equal within noise and meaningfully cheaper to serve")
    if res.promote and better:
        res.reasons.append("significantly better on at least one key metric, no regression beyond noise")
    return res
