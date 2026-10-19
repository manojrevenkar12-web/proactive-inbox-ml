"""Text features shared by training (PyTorch) and serving (numpy + ONNX Runtime).

Hashed word uni/bigrams and character 3-5-grams, fastText style: no vocabulary to ship, robust
to typos and unseen words, and the exact same code runs in training and in the slim inference
image. Collisions are rare at 2**18 buckets and the model learns around them.
"""

from __future__ import annotations

import re
import zlib

import numpy as np

BUCKETS = 2**18
_TOKEN = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")


def normalise(text: str) -> str:
    return text.lower().strip()


def tokens(text: str) -> list[str]:
    return _TOKEN.findall(normalise(text))


def _h(s: str) -> int:
    return zlib.crc32(s.encode()) % (BUCKETS - 1) + 1  # 0 is reserved for padding / empty


def ngram_ids(text: str, char_ngrams: bool = True) -> list[int]:
    toks = tokens(text)
    ids = [_h("w:" + t) for t in toks]
    ids += [_h(f"b:{a}_{b}") for a, b in zip(toks, toks[1:], strict=False)]
    if char_ngrams:
        for t in toks:
            w = f"<{t}>"
            for n in (3, 4, 5):
                ids += [_h(f"c{n}:" + w[i : i + n]) for i in range(len(w) - n + 1)]
    return ids or [0]


def batch(texts: list[str], char_ngrams: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Flat ids + offsets, the input format of nn.EmbeddingBag and of the exported ONNX models."""
    all_ids: list[int] = []
    offsets: list[int] = []
    for t in texts:
        offsets.append(len(all_ids))
        all_ids.extend(ngram_ids(t, char_ngrams))
    return np.asarray(all_ids, dtype=np.int64), np.asarray(offsets, dtype=np.int64)


MAX_IDS = 384


def padded(texts: list[str], char_ngrams: bool = True, max_ids: int = MAX_IDS) -> np.ndarray:
    """[batch, length] int64 ids, 0 = padding. The format both PyTorch and ONNX models take."""
    rows = [ngram_ids(t, char_ngrams)[:max_ids] for t in texts]
    width = max(len(r) for r in rows) if rows else 1
    out = np.zeros((len(rows), width), dtype=np.int64)
    for i, r in enumerate(rows):
        out[i, : len(r)] = r
    return out


SEQ_BUCKETS = 2**16
SEQ_LEN = 40
CLS = 1


def sequence(texts: list[str], max_len: int = SEQ_LEN) -> np.ndarray:
    """[batch, length] word-order-preserving ids for the transformer: [CLS] + hashed words; 0 = padding."""
    rows = [[CLS] + [zlib.crc32(("s:" + t).encode()) % (SEQ_BUCKETS - 2) + 2 for t in tokens(x)][: max_len - 1] for x in texts]
    width = max(len(r) for r in rows) if rows else 1
    out = np.zeros((len(rows), width), dtype=np.int64)
    for i, r in enumerate(rows):
        out[i, : len(r)] = r
    return out
