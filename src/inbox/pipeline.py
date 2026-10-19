"""Training pipeline: data → models → evaluation → release gate → registry → serving bundle.

`inbox train` runs everything and writes artifacts/ (models, registry, reports). `--fast` shrinks
every stage so CI can exercise the whole path in a few minutes; numbers in the README come from a
full run.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import shutil
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .config import home, models_dir, reports_dir
from .data import clinc
from .evaluation import classification as cls_eval
from .evaluation import ranking
from .evaluation.gate import compare
from .evaluation.registry import Registry


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _items(pred: list[str], gold: list[str]) -> dict[str, np.ndarray]:
    it = cls_eval.per_item(pred, gold)
    return {"inscope_accuracy": it["inscope_correct"], "oos_recall": it["oos_detected"]}


def _subset(split: clinc.Split, frac: float, seed: int = 0) -> clinc.Split:
    if frac >= 1:
        return split
    rng = np.random.default_rng(seed)
    by_label: dict[str, list[int]] = {}
    for i, lab in enumerate(split.labels):
        by_label.setdefault(lab, []).append(i)
    keep = sorted(i for idxs in by_label.values() for i in rng.choice(idxs, max(1, int(len(idxs) * frac)), replace=False))
    return clinc.Split([split.texts[i] for i in keep], [split.labels[i] for i in keep])


# --------------------------------------------------------------------------- classifier


def train_classifier(fast: bool, with_transformer: bool, log=print) -> dict:
    import torch

    from .classify import baseline, export, predictor
    from .classify import calibrate as cal
    from .classify import model as M

    torch.set_num_threads(max(1, os.cpu_count() or 1))
    d = clinc.load()
    dom = clinc.domains()
    data_sha = _sha(clinc.ensure() / "data_full.json")
    train = _subset(d["train"], 0.3 if fast else 1.0)
    test = d["test"]
    out = models_dir() / "classifier"
    out.mkdir(parents=True, exist_ok=True)
    registry = Registry(models_dir() / "registry.json")
    results: dict = {"data": {"train": len(train.texts), "val": len(d["val"].texts), "test": len(test.texts), "sha256": data_sha}}

    def calibrated(logits_val: np.ndarray, labels: list[str]) -> cal.Decision:
        oi = labels.index(clinc.OOS)
        yv = np.array([-1 if lab == clinc.OOS else labels.index(lab) for lab in d["val"].labels])
        t = cal.fit_temperature(logits_val[yv >= 0], yv[yv >= 0])
        dec, _ = cal.choose_threshold(cal.softmax(logits_val, t), yv, oi, t)
        return dec

    def to_labels(probs: np.ndarray, dec: cal.Decision, labels: list[str]) -> list[str]:
        return [clinc.OOS if i < 0 else labels[i] for i in cal.decide(probs, dec)]

    def dom_of(label: str) -> str:
        return dom.get(label, clinc.OOS)

    candidates = []

    # 1. baseline: TF-IDF + logistic regression
    log("  classifier: TF-IDF + logistic regression baseline")
    t0 = time.perf_counter()
    base = baseline.build()
    base.fit(train.texts, train.labels)
    labels_b = list(base.classes_)
    dec_b = calibrated(baseline.logits(base, d["val"].texts), labels_b)
    pred_b = to_labels(cal.softmax(baseline.logits(base, test.texts), dec_b.temperature), dec_b, labels_b)
    lat_b = predictor.latency(lambda x: base.decision_function(x), test.texts[:500])
    size_b = len(pickle.dumps(base)) / 1e6
    candidates.append(
        {
            "name": "tfidf_logreg",
            "pred": pred_b,
            "latency": lat_b,
            "size_mb": round(size_b, 1),
            "train_s": round(time.perf_counter() - t0, 1),
            "artifact": None,
        }
    )

    # 2. n-gram neural classifier → ONNX → int8
    log("  classifier: hashed n-gram network (PyTorch) → ONNX int8")
    cfg = M.TrainConfig(epochs=6 if fast else 20)
    r = M.train(train.texts, train.labels, cfg)
    dec_n = calibrated(M.logits(r.model, d["val"].texts), r.labels)
    fp32, int8 = out / "ngram_fp32.onnx", out / "ngram_int8.onnx"
    export.to_onnx(r.model, fp32)
    export.quantise(fp32, int8)
    meta = {
        "labels": r.labels,
        "temperature": dec_n.temperature,
        "threshold": dec_n.threshold,
        "version": "",
        "char_ngrams": True,
        "arch": "ngram",
        "train_config": asdict(cfg),
    }
    for path, name in [(fp32, "ngram_fp32"), (int8, "ngram_int8")]:
        meta["version"] = name
        (out / f"{name}.json").write_text(json.dumps(meta))
        clf = predictor.ActionClassifier(path, out / f"{name}.json")
        preds = [x["intent"] for x in clf.predict(test.texts)]
        candidates.append(
            {
                "name": name,
                "pred": preds,
                "latency": predictor.latency(clf.predict, test.texts[:1000]),
                "latency_b64": predictor.latency(clf.predict, test.texts, batch=64),
                "size_mb": round(path.stat().st_size / 1e6, 1),
                "train_s": r.seconds,
                "artifact": str(path),
            }
        )

    # 3. optional challenger: a small transformer trained from scratch
    if with_transformer:
        log("  classifier: 2-layer transformer challenger (slow on CPU)")
        tcfg = M.TrainConfig(arch="transformer", epochs=4 if fast else 15, lr_head=1e-3, dropout=0.1)
        rt = M.train(train.texts, train.labels, tcfg)
        dec_t = calibrated(M.logits(rt.model, d["val"].texts), rt.labels)
        tpath = out / "transformer.onnx"
        export.to_onnx(rt.model, tpath)
        meta_t = {
            "labels": rt.labels,
            "temperature": dec_t.temperature,
            "threshold": dec_t.threshold,
            "version": "transformer",
            "arch": "transformer",
        }
        (out / "transformer.json").write_text(json.dumps(meta_t))
        clf_t = predictor.ActionClassifier(tpath, out / "transformer.json")
        preds = [x["intent"] for x in clf_t.predict(test.texts)]
        candidates.append(
            {
                "name": "transformer",
                "pred": preds,
                "latency": predictor.latency(clf_t.predict, test.texts[:1000]),
                "size_mb": round(tpath.stat().st_size / 1e6, 1),
                "train_s": rt.seconds,
                "artifact": str(tpath),
            }
        )

    # 4. evaluation and the release gate, in order: each candidate challenges the current champion
    for c in candidates:
        c["report"] = cls_eval.report(c["pred"], test.labels, dom)
        c["items"] = _items(c["pred"], test.labels)
        c["slice_items"] = {
            dom: np.array([pp == gg for pp, gg, dd in zip(c["pred"], test.labels, [dom_of(x) for x in test.labels], strict=True) if dd == dom])
            for dom in sorted(set(dom.values()) - {clinc.OOS})
        }
        c["latency_p95_ms"] = c["latency"]["p95_ms"]
    results["candidates"] = {}
    champion = None
    for c in candidates:
        metrics = {**c["report"], "latency_ms_batch1": c["latency"], "size_mb": c["size_mb"], "train_seconds": c["train_s"]}
        if "latency_b64" in c:
            metrics["latency_ms_batch64"] = c["latency_b64"]
        if champion is None:
            decision, reasons, deltas = "champion", ["first model: becomes the baseline champion"], {}
        else:
            g = compare(champion, c)
            decision, reasons, deltas = ("champion" if g.promote else "rejected"), g.reasons, g.deltas
        registry.add("action_classifier", c["name"], c["artifact"], metrics, data_sha, decision, reasons)
        results["candidates"][c["name"]] = {"metrics": metrics, "decision": decision, "reasons": reasons, "deltas_vs_champion": deltas}
        log(
            f"    {c['name']:<14} in-scope {c['report']['inscope_accuracy']:.4f}  oos recall {c['report']['oos_recall']:.4f}  p95 {c['latency_p95_ms']:.3f} ms  → {decision}"
        )
        if decision == "champion":
            champion = c
    # serving bundle: the champion (only ONNX models can be served by the slim image)
    promoted = [c for c in candidates if results["candidates"][c["name"]]["decision"] == "champion" and c["artifact"]]
    if promoted:
        served = promoted[-1]
    elif fast:  # a smoke run on a fraction of the data: serve the int8 model anyway, flagged
        served = next(c for c in candidates if c["name"] == "ngram_int8")
        log("    no ONNX model passed the gate in this fast run; serving ngram_int8 unpromoted")
    else:
        raise RuntimeError("no servable model passed the release gate")
    shutil.copy(served["artifact"], out / "serving.onnx")
    serving_meta = json.loads((out / f"{served['name']}.json").read_text())
    # Reference distribution for drift monitoring in production: confidence histogram and
    # out-of-scope rate of the served model on the validation split.
    ref = predictor.ActionClassifier(Path(served["artifact"]), out / f"{served['name']}.json").predict(d["val"].texts)
    conf = np.array([x["confidence"] for x in ref if not x["out_of_scope"]])
    serving_meta["reference"] = {
        "oos_rate": round(float(np.mean([x["out_of_scope"] for x in ref])), 4),
        "confidence_hist": np.histogram(conf, bins=10, range=(0, 1))[0].tolist(),
    }
    serving_meta["gate_status"] = results["candidates"][served["name"]]["decision"]
    (out / "serving.json").write_text(json.dumps(serving_meta))
    results["served"] = served["name"]

    # 5. learning curve: how much labelled data does this task need?
    curve = []
    for frac in [0.25, 1.0] if fast else [0.1, 0.25, 0.5, 1.0]:
        sub = _subset(d["train"], frac, seed=3)
        rr = M.train(sub.texts, sub.labels, M.TrainConfig(epochs=6 if fast else 20))
        dd = calibrated(M.logits(rr.model, d["val"].texts), rr.labels)
        pr = to_labels(cal.softmax(M.logits(rr.model, test.texts), dd.temperature), dd, rr.labels)
        rep = cls_eval.report(pr, test.labels)
        curve.append(
            {"train_examples_per_intent": int(round(100 * frac)), "inscope_accuracy": rep["inscope_accuracy"], "oos_recall": rep["oos_recall"]}
        )
    results["learning_curve"] = curve
    return results


# --------------------------------------------------------------------------- retrieval


def train_retrieval(fast: bool, log=print) -> dict:
    import torch

    from .retrieval import corpus, rerank
    from .retrieval import encoder as En
    from .retrieval.index import UserIndex

    torch.set_num_threads(max(1, os.cpu_count() or 1))
    sp = corpus.splits()
    if fast:
        sp = {
            "encoder": corpus.generate(60, corpus.ENCODER, 7),
            "rerank": corpus.generate(20, corpus.RERANK, 8),
            "valid": corpus.generate(10, corpus.RERANK, 9),
            "test": corpus.generate(15, corpus.TEST, 10),
        }
    log(f"  retrieval: dual encoder on {len(sp['encoder'].queries):,} questions")
    model, info = En.train(sp["encoder"], epochs=2 if fast else 6)
    out = models_dir() / "retrieval"
    out.mkdir(parents=True, exist_ok=True)
    En.to_onnx(model, out / "encoder.onnx")

    def embed(texts: list[str]) -> np.ndarray:
        return En.embed(model, texts)

    idx = {k: {u: UserIndex(m, embed) for u, m in c.messages.items()} for k, c in sp.items() if k != "encoder"}
    res: dict = {
        "encoder": info,
        "corpus": {
            k: {"users": len(c.messages), "messages": sum(len(m) for m in c.messages.values()), "questions": len(c.queries)} for k, c in sp.items()
        },
    }
    test = sp["test"]
    for mode in ["bm25", "dense", "hybrid"]:
        runs = [([x.message.id for x in idx["test"][q.user].search(q.text, 50, mode)], q.relevance) for q in test.queries]
        res[mode] = ranking.summary(runs)
    # The same encoder on NEW users written with the phrasing it was trained on: the gap to the test
    # numbers shows how much of its skill is memorised wording rather than meaning.
    seen = corpus.generate(10 if fast else 30, corpus.ENCODER, seed=99)
    seen_idx = {u: UserIndex(m, embed) for u, m in seen.messages.items()}
    res["dense_seen_wording"] = ranking.summary(
        [([x.message.id for x in seen_idx[q.user].search(q.text, 50, "dense")], q.relevance) for q in seen.queries]
    )
    log("  retrieval: LightGBM LambdaRank reranker")
    rr = rerank.train(sp["rerank"], idx["rerank"], sp["valid"], idx["valid"])
    rr.booster_.save_model(str(out / "reranker.txt"))
    t_search, t_rerank, runs = [], [], []
    for q in test.queries:
        t0 = time.perf_counter()
        cands = idx["test"][q.user].search(q.text, 50, "candidates")
        t1 = time.perf_counter()
        ranked = rerank.rerank(rr, q.text, cands)
        t_search.append((t1 - t0) * 1000)
        t_rerank.append((time.perf_counter() - t1) * 1000)
        runs.append(([x.message.id for x in ranked], q.relevance))
    res["candidates_plus_reranker"] = ranking.summary(runs)
    res["latency_ms"] = {
        "first_stage_p50": round(float(np.percentile(t_search, 50)), 2),
        "first_stage_p95": round(float(np.percentile(t_search, 95)), 2),
        "rerank_p50": round(float(np.percentile(t_rerank, 50)), 2),
        "rerank_p95": round(float(np.percentile(t_rerank, 95)), 2),
    }
    res["reranker_feature_importance"] = {k: int(v) for k, v in zip(rerank.FEATURES, rr.feature_importances_, strict=True)}
    res["by_kind"] = {}
    for kind in sorted({q.kind for q in test.queries}):
        sel = [run for run, q in zip(runs, test.queries, strict=True) if q.kind == kind]
        res["by_kind"][kind] = ranking.summary(sel)["ndcg@10"]
    log(
        f"    nDCG@10 test (unseen wording): BM25 {res['bm25']['ndcg@10']:.3f}  dense {res['dense']['ndcg@10']:.3f}  hybrid {res['hybrid']['ndcg@10']:.3f}  "
        f"+ reranker {res['candidates_plus_reranker']['ndcg@10']:.3f}"
    )
    return res


# --------------------------------------------------------------------------- personalisation


def train_personalisation(fast: bool, log=print) -> dict:
    from .personalize import models as Mo
    from .personalize import sim

    n_users = 400 if fast else 2000
    world = sim.make_world(n_users)
    ev = sim.log_events(world, 1)
    rng = np.random.default_rng(99)
    tu = np.repeat(np.arange(n_users), 20)
    tx = sim.contexts(rng, len(tu))
    p_true = world.p_accept(tu, tx)
    oracle = float(p_true.max(1).mean())
    bucket = np.where(world.n_events[tu] < 15, "cold", np.where(world.n_events[tu] < 60, "warm", "heavy"))
    held = sim.log_events(world, 2, per_user=np.full(n_users, 10))

    def value(scores: np.ndarray) -> dict:
        chosen = p_true[np.arange(len(tu)), scores.argmax(1)]
        return {
            "acceptance": round(float(chosen.mean()), 4),
            "share_of_oracle": round(float(chosen.mean()) / oracle, 4),
            **{f"acceptance_{b}": round(float(chosen[bucket == b].mean()), 4) for b in ("cold", "warm", "heavy")},
        }

    def offline_auc(p: Mo.Params, theta: np.ndarray) -> float:
        s = Mo.scores(p, held.ctx, theta[held.user])[np.arange(len(held.user)), held.action]
        return round(Mo.auc(held.accepted, s), 4)

    res: dict = {
        "users": n_users,
        "logged_events": int(len(ev.user)),
        "median_events_per_user": float(np.median(world.n_events)),
        "oracle_acceptance": round(oracle, 4),
        "random_acceptance": round(float(p_true.mean()), 4),
        "models": {},
    }
    zeros = np.zeros((n_users, sim.TASTE_DIM))
    res["models"]["popularity"] = {**value(np.tile(Mo.popularity(ev), (len(tu), 1))), "raw_events_leave_device": True}
    log("  personalisation: global and central personalised models")
    g, _ = Mo.fit_central(ev, n_users, personalised=False, steps=200 if fast else 600)
    res["models"]["global_contextual"] = {**value(Mo.scores(g, tx, zeros[tu])), "auc": offline_auc(g, zeros), "raw_events_leave_device": True}
    c, th = Mo.fit_central(ev, n_users, personalised=True, steps=200 if fast else 600)
    res["models"]["central_personalised"] = {**value(Mo.scores(c, tx, th[tu])), "auc": offline_auc(c, th), "raw_events_leave_device": True}
    serving = None
    for sigma in (0.0, 1.0):
        log(f"  personalisation: federated, noise multiplier {sigma}")
        cfg = Mo.FedConfig(noise_multiplier=sigma, rounds=60 if fast else 300)
        f, info = Mo.fit_federated(ev, n_users, cfg)
        thf = Mo.personalise_on_device(f, ev, n_users)
        name = "federated" if sigma == 0 else "federated_dp"
        res["models"][name] = {
            **value(Mo.scores(f, tx, thf[tu])),
            "auc": offline_auc(f, thf),
            "raw_events_leave_device": False,
            "epsilon": info["epsilon"],
            "delta": round(info["delta"], 6),
            "noise_multiplier": sigma,
            "rounds": cfg.rounds,
            "devices_per_round": round(cfg.sample_rate * n_users),
            "clip": cfg.clip,
            "clipped_share": info["clipped_share"],
        }
        if sigma > 0:
            serving = f
    out = models_dir() / "personalisation"
    out.mkdir(parents=True, exist_ok=True)
    (out / "global.json").write_text(
        json.dumps(
            {
                "actions": sim.ACTIONS,
                "message_types": sim.MESSAGE_TYPES,
                "senders": sim.SENDERS,
                "times": sim.TIMES,
                "b": serving.b.tolist(),
                "w": serving.w.tolist(),
                "v": serving.v.tolist(),
                "model": "federated_dp",
                "epsilon": res["models"]["federated_dp"]["epsilon"],
            }
        )
    )
    for k, m in res["models"].items():
        log(f"    {k:<22} acceptance {m['acceptance']:.3f} ({m['share_of_oracle']:.1%} of oracle)")
    return res


# --------------------------------------------------------------------------- all


def run(fast: bool = False, with_transformer: bool = False, log=print) -> dict:
    t0 = time.perf_counter()
    home().mkdir(parents=True, exist_ok=True)
    reg = models_dir() / "registry.json"
    if reg.exists():
        reg.unlink()
    log("Training the action classifier")
    report = {"fast": fast, "classifier": train_classifier(fast, with_transformer, log)}
    log("Training context retrieval")
    report["retrieval"] = train_retrieval(fast, log)
    log("Training personalised suggestions")
    report["personalisation"] = train_personalisation(fast, log)
    report["total_seconds"] = round(time.perf_counter() - t0, 1)
    reports_dir().mkdir(parents=True, exist_ok=True)
    (reports_dir() / "report.json").write_text(json.dumps(report, indent=2, default=str))
    log(f"Done in {report['total_seconds']}s → {reports_dir() / 'report.json'}")
    return report


def dp_scale(populations: tuple[int, ...] = (2000, 10000), sigma: float = 1.0, log=print) -> dict:
    """How the cost of differential privacy changes with the number of users.

    The DP noise is added once per round to the sum of device updates; with more devices per
    round the same noise is divided by a larger cohort, so the utility cost should shrink while
    ε stays the same (same sampling rate, rounds and noise multiplier)."""
    from .personalize import models as Mo
    from .personalize import sim

    out = {}
    for n in populations:
        world = sim.make_world(n, seed=11)
        ev = sim.log_events(world, 1)
        rng = np.random.default_rng(99)
        tu = np.repeat(np.arange(n), 10)
        tx = sim.contexts(rng, len(tu))
        p = world.p_accept(tu, tx)
        oracle = p.max(1).mean()

        def share(s: np.ndarray, p: np.ndarray = p, oracle: float = oracle) -> float:
            return round(float(p[np.arange(len(p)), s.argmax(1)].mean() / oracle), 4)

        g, _ = Mo.fit_central(ev, n, personalised=False)
        row = {"global_contextual": share(Mo.scores(g, tx, np.zeros((len(tu), sim.TASTE_DIM))))}
        for s in (0.0, sigma):
            f, info = Mo.fit_federated(ev, n, Mo.FedConfig(noise_multiplier=s))
            th = Mo.personalise_on_device(f, ev, n)
            row["federated_dp" if s else "federated"] = share(Mo.scores(f, tx, th[tu]))
            if s:
                row["epsilon"] = info["epsilon"]
                row["delta"] = round(info["delta"], 7)
        out[str(n)] = row
        log(f"  {n:>6} users: {row}")
    reports_dir().mkdir(parents=True, exist_ok=True)
    (reports_dir() / "dp_scale.json").write_text(json.dumps(out, indent=2))
    return out
