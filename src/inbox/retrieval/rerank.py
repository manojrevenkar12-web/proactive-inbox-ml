"""Learned reranker: LightGBM LambdaRank over the hybrid top-50.

First-stage retrieval scores each message on its own. The reranker adds what a person would use:
whether the contact named in the question sent the message, how recent it is among similar
candidates, and whether it is a follow-up to a strong candidate from the same contact (the
"actually, make it 8 instead" message shares almost no words with the question, but it follows
the plan it corrects). It is trained on different users from the dense encoder, so its inputs
are not optimistically biased.
"""

from __future__ import annotations

from datetime import timedelta

import lightgbm as lgb
import numpy as np

from .. import features
from .corpus import Corpus
from .index import Candidate, UserIndex

FEATURES = [
    "bm25",
    "bm25_rank",
    "dense",
    "dense_rank",
    "rrf",
    "contact_named",
    "recency_rank",
    "follows_strong_same_contact",
    "minutes_after_strong",
    "has_number",
    "length",
    "query_overlap",
]
CORRECTION_WINDOW = timedelta(days=3)


def featurise(query: str, cands: list[Candidate]) -> np.ndarray:
    q_tokens = set(features.tokens(query))
    newest_first = sorted(range(len(cands)), key=lambda i: cands[i].message.sent_at, reverse=True)
    recency = {i: r for r, i in enumerate(newest_first)}
    # "strong" = top-3 by first-stage fusion score
    strong = sorted(cands, key=lambda c: -c.rrf)[:3]
    rows = []
    for i, c in enumerate(cands):
        m = c.message
        named = float(m.contact.lower() in q_tokens)
        follows = [s for s in strong if s.message.contact == m.contact and s.message.sent_at < m.sent_at <= s.message.sent_at + CORRECTION_WINDOW]
        mins = min(((m.sent_at - s.message.sent_at).total_seconds() / 60 for s in follows), default=-1.0)
        toks = set(features.tokens(m.text))
        rows.append(
            [
                c.bm25,
                c.bm25_rank,
                c.dense,
                c.dense_rank,
                c.rrf,
                named,
                recency[i],
                float(bool(follows)),
                mins,
                float(any(ch.isdigit() for ch in m.text)),
                len(m.text),
                len(q_tokens & toks) / max(1, len(q_tokens)),
            ]
        )
    return np.asarray(rows, dtype=np.float32)


def build_dataset(c: Corpus, indexes: dict[str, UserIndex], k: int = 50):
    xs, ys, groups = [], [], []
    for q in c.queries:
        cands = indexes[q.user].search(q.text, k, mode="candidates")
        xs.append(featurise(q.text, cands))
        ys.append([q.relevance.get(x.message.id, 0) for x in cands])
        groups.append(len(cands))
    return np.vstack(xs), np.concatenate(ys), groups


def train(c_train: Corpus, idx_train: dict[str, UserIndex], c_valid: Corpus, idx_valid: dict[str, UserIndex], seed: int = 5) -> lgb.LGBMRanker:
    x, y, g = build_dataset(c_train, idx_train)
    xv, yv, gv = build_dataset(c_valid, idx_valid)
    model = lgb.LGBMRanker(
        objective="lambdarank",
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.9,
        random_state=seed,
        verbose=-1,
    )
    model.fit(x, y, group=g, eval_X=(xv,), eval_y=(yv,), eval_group=[gv], eval_at=[10], callbacks=[lgb.early_stopping(30, verbose=False)])
    return model


def rerank(model, query: str, cands: list[Candidate]) -> list[Candidate]:
    if not cands:
        return cands
    scores = model.predict(featurise(query, cands))
    return [cands[i] for i in np.argsort(-scores, kind="stable")]
