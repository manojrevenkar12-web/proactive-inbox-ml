"""PyTorch classifier: hashed n-gram embeddings, masked mean, one hidden layer.

Small on purpose. It answers in well under a millisecond on one CPU core, trains in a couple
of minutes, quantises to int8 with almost no loss, and needs no pretrained weights — so the
whole system can be retrained from labelled data alone, on the user's own data domain.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from .. import features


class NgramClassifier(nn.Module):
    def __init__(self, n_classes: int, dim: int = 64, hidden: int = 256, buckets: int = features.BUCKETS, dropout: float = 0.2):
        super().__init__()
        self.emb = nn.Embedding(buckets, dim, padding_idx=0, sparse=True)
        self.head = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, n_classes))
        nn.init.normal_(self.emb.weight, std=0.05)
        with torch.no_grad():
            self.emb.weight[0].zero_()

    def encode(self, ids: torch.Tensor) -> torch.Tensor:
        mask = (ids > 0).unsqueeze(-1).to(self.emb.weight.dtype)
        summed = (self.emb(ids) * mask).sum(1)
        return summed / mask.sum(1).clamp(min=1.0)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:  # logits
        return self.head(self.encode(ids))


class _Block(nn.Module):
    """Pre-norm transformer block with explicit attention (exports cleanly to ONNX)."""

    def __init__(self, dim: int, heads: int, dropout: float):
        super().__init__()
        self.heads = heads
        self.n1, self.n2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.qkv, self.proj = nn.Linear(dim, 3 * dim), nn.Linear(dim, dim)
        self.ff = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, key_mask: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        q, k, v = self.qkv(self.n1(x)).reshape(b, n, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4).unbind(0)
        att = (q @ k.transpose(-2, -1)) / (d // self.heads) ** 0.5
        att = att.masked_fill(~key_mask[:, None, None, :], -1e4).softmax(-1)
        x = x + self.drop(self.proj((att @ v).transpose(1, 2).reshape(b, n, d)))
        return x + self.drop(self.ff(self.n2(x)))


class TinyTransformer(nn.Module):
    """2-layer transformer encoder trained from scratch on hashed words (no pretrained weights available offline)."""

    def __init__(self, n_classes: int, dim: int = 128, layers: int = 2, heads: int = 4, dropout: float = 0.1, buckets: int = features.SEQ_BUCKETS):
        super().__init__()
        self.tok = nn.Embedding(buckets, dim, padding_idx=0)
        self.pos = nn.Embedding(features.SEQ_LEN, dim)
        self.blocks = nn.ModuleList([_Block(dim, heads, dropout) for _ in range(layers)])
        self.norm = nn.LayerNorm(dim)
        self.out = nn.Linear(dim, n_classes)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        mask = ids > 0
        pos = torch.arange(ids.shape[1], device=ids.device).unsqueeze(0)
        x = self.tok(ids) + self.pos(pos)
        for blk in self.blocks:
            x = blk(x, mask)
        return self.out(self.norm(x[:, 0]))


@dataclass
class TrainConfig:
    epochs: int = 20
    batch_size: int = 64
    lr_emb: float = 0.05
    lr_head: float = 2e-3
    dim: int = 64
    hidden: int = 256
    dropout: float = 0.2
    seed: int = 13
    label_smoothing: float = 0.05
    arch: str = "ngram"  # "ngram" | "transformer"
    char_ngrams: bool = True


@dataclass
class TrainResult:
    model: nn.Module
    labels: list[str]
    history: list[dict] = field(default_factory=list)
    seconds: float = 0.0


def train(texts: list[str], labels: list[str], cfg: TrainConfig | None = None, val: tuple[list[str], list[str]] | None = None) -> TrainResult:
    cfg = cfg or TrainConfig()
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    classes = sorted(set(labels))
    index = {c: i for i, c in enumerate(classes)}
    y = np.array([index[c] for c in labels])
    if cfg.arch == "transformer":
        ids = [list(r[r > 0]) for r in features.sequence(texts)]
        model: nn.Module = TinyTransformer(len(classes), dropout=cfg.dropout)
        opt_emb = None
        opt_head = torch.optim.AdamW(model.parameters(), lr=cfg.lr_head, weight_decay=0.01)
    else:
        ids = [features.ngram_ids(t, cfg.char_ngrams)[: features.MAX_IDS] for t in texts]
        model = NgramClassifier(len(classes), cfg.dim, cfg.hidden, dropout=cfg.dropout)
        opt_emb = torch.optim.SparseAdam(list(model.emb.parameters()), lr=cfg.lr_emb)
        opt_head = torch.optim.AdamW(model.head.parameters(), lr=cfg.lr_head, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    t0 = time.perf_counter()
    history = []
    steps_per_epoch = (len(ids) + cfg.batch_size - 1) // cfg.batch_size
    sched = torch.optim.lr_scheduler.OneCycleLR(opt_head, max_lr=cfg.lr_head, total_steps=cfg.epochs * steps_per_epoch)
    for epoch in range(cfg.epochs):
        model.train()
        order = rng.permutation(len(ids))
        total = 0.0
        for s in range(0, len(order), cfg.batch_size):
            b = order[s : s + cfg.batch_size]
            width = max(len(ids[i]) for i in b)
            x = np.zeros((len(b), width), dtype=np.int64)
            for r, i in enumerate(b):
                x[r, : len(ids[i])] = ids[i]
            logits = model(torch.from_numpy(x))
            loss = loss_fn(logits, torch.from_numpy(y[b]))
            if opt_emb:
                opt_emb.zero_grad()
            opt_head.zero_grad()
            loss.backward()
            if opt_emb:
                opt_emb.step()
            opt_head.step()
            sched.step()
            total += loss.item() * len(b)
        row = {"epoch": epoch + 1, "loss": round(total / len(ids), 4)}
        if val:
            pred = predict_labels(model, classes, val[0], cfg.char_ngrams)
            row["val_acc"] = round(float(np.mean([p == t for p, t in zip(pred, val[1], strict=True)])), 4)
        history.append(row)
    return TrainResult(model, classes, history, round(time.perf_counter() - t0, 1))


def encode_inputs(model: nn.Module, texts: list[str], char_ngrams: bool = True) -> np.ndarray:
    return features.sequence(texts) if isinstance(model, TinyTransformer) else features.padded(texts, char_ngrams)


@torch.no_grad()
def logits(model: nn.Module, texts: list[str], char_ngrams: bool = True, batch_size: int = 512) -> np.ndarray:
    model.eval()
    out = [model(torch.from_numpy(encode_inputs(model, texts[s : s + batch_size], char_ngrams))).numpy() for s in range(0, len(texts), batch_size)]
    return np.concatenate(out)


def predict_labels(model: nn.Module, classes: list[str], texts: list[str], char_ngrams: bool = True) -> list[str]:
    return [classes[i] for i in logits(model, texts, char_ngrams).argmax(1)]
