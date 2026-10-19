"""Per-user hybrid index: BM25 + dense vectors, fused with reciprocal rank fusion.

One index per user, by design: a query can only ever see its own user's messages, which is the
simplest possible guarantee against cross-user leakage, and keeps each index small enough to
search exhaustively in well under a millisecond.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from rank_bm25 import BM25Okapi

from .. import features
from .corpus import Message
from .encoder import doc_text

RRF_K = 60


@dataclass
class Candidate:
    message: Message
    bm25: float
    bm25_rank: int
    dense: float
    dense_rank: int
    rrf: float


class UserIndex:
    def __init__(self, messages: list[Message], embed: Callable[[list[str]], np.ndarray]):
        self.messages = messages
        self.docs = [doc_text(m.contact, m.text) for m in messages]
        self.bm25 = BM25Okapi([features.tokens(d) or ["_"] for d in self.docs])
        self.vectors = embed(self.docs)
        self.embed = embed

    def search(self, query: str, k: int = 50, mode: str = "hybrid") -> list[Candidate]:
        bm = self.bm25.get_scores(features.tokens(query) or ["_"])
        dn = self.vectors @ self.embed([query])[0]
        bm_rank = np.empty(len(bm), dtype=int)
        bm_rank[np.argsort(-bm, kind="stable")] = np.arange(len(bm))
        dn_rank = np.empty(len(dn), dtype=int)
        dn_rank[np.argsort(-dn, kind="stable")] = np.arange(len(dn))
        rrf = 1 / (RRF_K + bm_rank + 1) + 1 / (RRF_K + dn_rank + 1)
        if mode == "bm25":
            top = np.argsort(bm_rank)[:k]
        elif mode == "dense":
            top = np.argsort(dn_rank)[:k]
        elif mode == "hybrid":
            top = np.argsort(-rrf, kind="stable")[:k]
        else:  # "candidates": union of dense top-k and BM25 top-k/2, the reranker's input
            pool = set(np.argsort(dn_rank)[:k]) | set(np.argsort(bm_rank)[: k // 2])
            top = sorted(pool, key=lambda i: -rrf[i])
        return [Candidate(self.messages[i], float(bm[i]), int(bm_rank[i]), float(dn[i]), int(dn_rank[i]), float(rrf[i])) for i in top]
