"""Dense retrieval encoder trained from scratch with a contrastive (InfoNCE) objective.

One shared tower for questions and messages: hashed n-gram embeddings, masked mean, a projection
and L2 normalisation. Batches mix in-batch negatives with hard negatives — messages of the same
kind from other contacts — which is what teaches the encoder that "dinner with Anna" must not
match Ben's dinner plan.
"""

from __future__ import annotations

import time

import numpy as np
import torch
from torch import nn

from .. import features
from .corpus import Corpus


def doc_text(contact: str, text: str) -> str:
    """What gets indexed: the sender is part of the document, as it is in any chat UI."""
    return f"{contact}: {text}"


class DualEncoder(nn.Module):
    def __init__(self, dim: int = 96, out: int = 128, buckets: int = features.BUCKETS):
        super().__init__()
        self.emb = nn.Embedding(buckets, dim, padding_idx=0, sparse=True)
        self.proj = nn.Sequential(nn.Linear(dim, out), nn.Tanh(), nn.Linear(out, out))
        nn.init.normal_(self.emb.weight, std=0.05)
        with torch.no_grad():
            self.emb.weight[0].zero_()

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        mask = (ids > 0).unsqueeze(-1).float()
        pooled = (self.emb(ids) * mask).sum(1) / mask.sum(1).clamp(min=1.0)
        return nn.functional.normalize(self.proj(pooled), dim=-1)


def training_pairs(c: Corpus, rng: np.random.Generator) -> list[tuple[str, str, str]]:
    """(question, positive document, hard negative document)."""
    by_id = {m.id: m for msgs in c.messages.values() for m in msgs}
    out = []
    for q in c.queries:
        same_kind = [m for m in c.messages[q.user] if m.kind == q.kind and m.id not in q.relevance]
        for mid in q.relevance:
            m = by_id[mid]
            neg = same_kind[rng.integers(len(same_kind))] if same_kind else c.messages[q.user][rng.integers(len(c.messages[q.user]))]
            out.append((q.text, doc_text(m.contact, m.text), doc_text(neg.contact, neg.text)))
    return out


def train(c: Corpus, epochs: int = 8, batch_size: int = 128, tau: float = 0.05, seed: int = 3) -> tuple[DualEncoder, dict]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    pairs = training_pairs(c, rng)
    model = DualEncoder()
    opt_e = torch.optim.SparseAdam(list(model.emb.parameters()), lr=0.03)
    opt_p = torch.optim.AdamW(model.proj.parameters(), lr=2e-3)
    t0 = time.perf_counter()
    losses = []
    for _ in range(epochs):
        model.train()
        order = rng.permutation(len(pairs))
        total = 0.0
        for s in range(0, len(order), batch_size):
            b = [pairs[i] for i in order[s : s + batch_size]]
            q = model(torch.from_numpy(features.padded([x[0] for x in b])))
            d = model(torch.from_numpy(features.padded([x[1] for x in b] + [x[2] for x in b])))
            scores = q @ d.T / tau  # [B, 2B]: positives on the diagonal, the rest are negatives
            loss = nn.functional.cross_entropy(scores, torch.arange(len(b)))
            opt_e.zero_grad()
            opt_p.zero_grad()
            loss.backward()
            opt_e.step()
            opt_p.step()
            total += loss.item() * len(b)
        losses.append(round(total / len(pairs), 4))
    return model, {"pairs": len(pairs), "epochs": epochs, "loss": losses, "seconds": round(time.perf_counter() - t0, 1)}


@torch.no_grad()
def embed(model: DualEncoder, texts: list[str], batch_size: int = 512) -> np.ndarray:
    model.eval()
    return np.concatenate([model(torch.from_numpy(features.padded(texts[s : s + batch_size]))).numpy() for s in range(0, len(texts), batch_size)])


def to_onnx(model: DualEncoder, path) -> None:
    model.eval()
    torch.onnx.export(
        model,
        (torch.ones((2, 16), dtype=torch.int64),),
        str(path),
        input_names=["ids"],
        output_names=["embedding"],
        dynamic_axes={"ids": {0: "batch", 1: "length"}, "embedding": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
