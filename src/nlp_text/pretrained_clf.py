"""Frozen MiniLM embeddings + LogisticRegression classifier head.

Leakage rules: encoder is pretrained (external) but **not** fit on this
dataset's test split; the logistic head is fit on **train embeddings only**.
Report metrics on OOS / test only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression

from nlp_text.pretrained_encoder import (
    DEFAULT_MODEL_NAME,
    PretrainedEncoder,
    build_pretrained_encoder,
)


@dataclass
class PretrainedLogReg:
    """Bundle: frozen encoder + fitted logistic head."""

    encoder: PretrainedEncoder
    clf: LogisticRegression
    model_name: str


def fit_pretrained_logreg(
    texts: Sequence[str],
    y: np.ndarray,
    *,
    encoder: PretrainedEncoder | None = None,
    model_name: str = DEFAULT_MODEL_NAME,
    C: float = 1.0,
    max_iter: int = 1000,
    random_state: int = 42,
    batch_size: int = 32,
) -> PretrainedLogReg:
    """Encode train texts with frozen MiniLM, fit LogisticRegression on train only."""
    enc = encoder or build_pretrained_encoder(model_name)
    X = enc.encode(texts, batch_size=batch_size)
    y = np.asarray(y, dtype=np.int64)
    clf = LogisticRegression(
        C=C,
        max_iter=max_iter,
        random_state=random_state,
        solver="lbfgs",
    )
    clf.fit(X, y)
    return PretrainedLogReg(encoder=enc, clf=clf, model_name=enc.model_name)


def predict_pretrained_logreg(
    model: PretrainedLogReg,
    texts: Sequence[str],
    *,
    batch_size: int = 32,
) -> np.ndarray:
    """Encode texts with the frozen encoder and predict with the fitted head."""
    X = model.encoder.encode(texts, batch_size=batch_size)
    return model.clf.predict(X)


def embed_train_test(
    encoder: PretrainedEncoder,
    train_texts: Sequence[str],
    test_texts: Sequence[str],
    *,
    batch_size: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    """Encode train and test separately (no joint fit — encoder is frozen)."""
    X_train = encoder.encode(train_texts, batch_size=batch_size)
    X_test = encoder.encode(test_texts, batch_size=batch_size)
    return X_train, X_test
