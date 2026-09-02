"""Confusion matrix and top misclassified examples."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.metrics import confusion_matrix


def compute_confusion(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    labels: Sequence[int] | None = None,
) -> np.ndarray:
    """Return sklearn confusion matrix (rows=true, cols=pred)."""
    return confusion_matrix(y_true, y_pred, labels=labels)


def top_misclassified(
    texts: Sequence[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    target_names: Sequence[str] | None = None,
    top_k: int = 10,
    snippet_len: int = 160,
) -> list[dict]:
    """Return top-k misclassified examples with true/pred labels and text snippet.

    Ordering is by index among errors (stable); no confidence scores required.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    texts = list(texts)
    err_idx = np.where(y_true != y_pred)[0]
    results: list[dict] = []
    for i in err_idx[:top_k]:
        true_lab = int(y_true[i])
        pred_lab = int(y_pred[i])
        true_name = (
            target_names[true_lab]
            if target_names is not None and true_lab < len(target_names)
            else str(true_lab)
        )
        pred_name = (
            target_names[pred_lab]
            if target_names is not None and pred_lab < len(target_names)
            else str(pred_lab)
        )
        snippet = " ".join(texts[i].split())
        if len(snippet) > snippet_len:
            snippet = snippet[: snippet_len - 3] + "..."
        results.append(
            {
                "index": int(i),
                "true": true_lab,
                "pred": pred_lab,
                "true_name": true_name,
                "pred_name": pred_name,
                "text": snippet,
            }
        )
    return results
