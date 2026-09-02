"""TF-IDF + LogisticRegression baseline (vectorizer fit on train only)."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def build_tfidf_logreg(
    *,
    min_df: int = 2,
    ngram_range: tuple[int, int] = (1, 2),
    max_features: int = 20_000,
    max_iter: int = 1000,
    C: float = 1.0,
    random_state: int = 42,
) -> Pipeline:
    """Return an unfitted sklearn Pipeline: TF-IDF → LogisticRegression."""
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    min_df=min_df,
                    ngram_range=ngram_range,
                    max_features=max_features,
                    lowercase=True,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=max_iter,
                    C=C,
                    random_state=random_state,
                    solver="lbfgs",
                ),
            ),
        ]
    )


def fit_baseline(
    texts: Sequence[str],
    y: np.ndarray,
    **kwargs,
) -> Pipeline:
    """Fit the TF-IDF + LR pipeline on train texts/labels only."""
    pipe = build_tfidf_logreg(**kwargs)
    pipe.fit(list(texts), np.asarray(y))
    return pipe


def predict_baseline(model: Pipeline, texts: Sequence[str]) -> np.ndarray:
    """Predict class labels for texts."""
    return model.predict(list(texts))
