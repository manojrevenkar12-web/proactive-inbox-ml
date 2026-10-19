"""TF-IDF + logistic regression: the baseline every neural model has to beat."""

from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline


def build(c: float = 20.0) -> Pipeline:
    feats = FeatureUnion(
        [
            ("word", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=2)),
        ]
    )
    return Pipeline([("features", feats), ("clf", LogisticRegression(C=c, max_iter=2000))])


def logits(pipe: Pipeline, texts: list[str]) -> np.ndarray:
    return pipe.decision_function(texts)
