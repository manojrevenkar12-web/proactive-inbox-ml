"""Ranking metrics with graded relevance."""

from __future__ import annotations

import numpy as np


def dcg(gains: list[float]) -> float:
    return float(sum((2**g - 1) / np.log2(i + 2) for i, g in enumerate(gains)))


def ndcg_at(ranked: list[str], rel: dict[str, int], k: int = 10) -> float:
    ideal = dcg(sorted(rel.values(), reverse=True)[:k])
    return dcg([rel.get(m, 0) for m in ranked[:k]]) / ideal if ideal else 0.0


def recall_at(ranked: list[str], rel: dict[str, int], k: int) -> float:
    """Was the best answer (highest grade) retrieved in the top k?"""
    best = {m for m, g in rel.items() if g == max(rel.values())}
    return float(bool(best & set(ranked[:k])))


def mrr_at(ranked: list[str], rel: dict[str, int], k: int = 10) -> float:
    best = {m for m, g in rel.items() if g == max(rel.values())}
    for i, m in enumerate(ranked[:k]):
        if m in best:
            return 1.0 / (i + 1)
    return 0.0


def summary(runs: list[tuple[list[str], dict[str, int]]]) -> dict:
    return {
        "ndcg@10": round(float(np.mean([ndcg_at(r, q) for r, q in runs])), 4),
        "mrr@10": round(float(np.mean([mrr_at(r, q) for r, q in runs])), 4),
        "recall@1": round(float(np.mean([recall_at(r, q, 1) for r, q in runs])), 4),
        "recall@10": round(float(np.mean([recall_at(r, q, 10) for r, q in runs])), 4),
        "recall@50": round(float(np.mean([recall_at(r, q, 50) for r, q in runs])), 4),
        "queries": len(runs),
    }
