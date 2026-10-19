import math

import numpy as np

from inbox import features
from inbox.classify import calibrate as cal
from inbox.evaluation import classification, ranking
from inbox.evaluation.gate import compare
from inbox.personalize import models as pm


def test_features_are_deterministic_and_padded():
    a, b = features.ngram_ids("Remind me at 5"), features.ngram_ids("remind me at 5")
    assert a == b and all(0 < i < features.BUCKETS for i in a)
    x = features.padded(["hi", "a much longer message about dinner"])
    assert x.shape[0] == 2 and (x[0] == 0).sum() > 0 and features.ngram_ids("") == [0]
    s = features.sequence(["hello world"])
    assert s[0, 0] == features.CLS and s.shape[1] == 3


def test_temperature_recovers_overconfidence():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 5, 4000)
    true_logits = rng.normal(0, 1, (4000, 5))
    true_logits[np.arange(4000), y] += 1.5
    t1 = cal.fit_temperature(true_logits, y)
    t = cal.fit_temperature(true_logits * 3.0, y)  # the same model, 3x over-confident
    assert abs(t / t1 - 3.0) < 0.15
    probs = cal.softmax(true_logits * 3.0, t)
    assert cal.ece(probs, y) < cal.ece(cal.softmax(true_logits * 3.0), y)


def test_threshold_respects_inscope_budget():
    rng = np.random.default_rng(1)
    n = 2000
    probs = rng.dirichlet(np.ones(4) * 0.1, n)
    y = probs.argmax(1).copy()
    y[:200] = -1  # out of scope rows
    probs[:200] = rng.dirichlet(np.ones(4) * 10, 200)  # flatter = less confident
    dec, curve = cal.choose_threshold(probs, y, None, 1.0, max_inscope_drop=0.02)
    chosen = next(c for c in curve if c["threshold"] == dec.threshold)
    assert chosen["inscope_accuracy"] >= curve[0]["inscope_accuracy"] - 0.02 and chosen["oos_recall"] > 0.5
    assert (cal.decide(probs, dec)[:200] == -1).mean() > 0.5


def test_classification_report_protocol():
    gold = ["a", "b", "oos", "oos", "a"]
    pred = ["a", "oos", "oos", "b", "a"]
    r = classification.report(pred, gold)
    assert r["inscope_accuracy"] == round(2 / 3, 4)  # predicting oos on an in-scope item is an error
    assert r["oos_recall"] == 0.5 and r["false_oos_rate"] == round(1 / 3, 4)


def test_ranking_metrics():
    rel = {"m1": 1, "m2": 2}
    assert ranking.ndcg_at(["m2", "m1", "x"], rel) == 1.0
    assert ranking.ndcg_at(["x", "m1", "m2"], rel) < 1.0
    assert ranking.mrr_at(["x", "m2"], rel) == 0.5 and ranking.recall_at(["m1"], rel, 1) == 0.0


def _cand(acc: float, oos: float, lat: float = 1.0, size: float = 10.0, n: int = 3000, seed: int = 0):
    rng = np.random.default_rng(seed)
    return {
        "items": {"inscope_accuracy": rng.random(n) < acc, "oos_recall": rng.random(1000) < oos},
        "slice_items": {"s": rng.random(400) < acc},
        "latency_p95_ms": lat,
        "size_mb": size,
    }


def test_gate_blocks_regressions_and_budgets():
    champ = _cand(0.90, 0.75, seed=1)
    assert not compare(champ, _cand(0.85, 0.75, seed=2)).promote
    assert compare(champ, _cand(0.93, 0.80, seed=3)).promote
    slow = compare(champ, _cand(0.93, 0.80, lat=9.0, seed=3))
    assert not slow.promote and any("latency" in r for r in slow.reasons)
    same = compare(champ, _cand(0.90, 0.75, seed=1))
    assert not same.promote  # not better, not cheaper
    assert compare(champ, _cand(0.90, 0.75, size=2.0, seed=1)).promote  # equal but 5x smaller


def test_privacy_accountant():
    # q = 1 reduces to the Gaussian mechanism: RDP(α) = α / (2σ²)
    assert math.isclose(pm._log_a_int(1.0, 2.0, 5) / 4, 5 / (2 * 4), rel_tol=1e-9)
    assert pm.epsilon(1.0, 0.05, 300, 1e-5) > pm.epsilon(2.0, 0.05, 300, 1e-5)
    assert pm.epsilon(1.0, 0.05, 600, 1e-5) > pm.epsilon(1.0, 0.05, 300, 1e-5)
    assert pm.epsilon(1.0, 0.01, 300, 1e-5) < pm.epsilon(1.0, 0.05, 300, 1e-5)
