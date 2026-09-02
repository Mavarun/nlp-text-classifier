"""Error analysis helpers."""

import numpy as np

from nlp_text.errors import compute_confusion, top_misclassified
from nlp_text.evaluate import compute_metrics


def test_confusion_and_misclassified():
    texts = ["a", "b", "c", "d", "e"]
    y_true = np.array([0, 1, 0, 1, 0])
    y_pred = np.array([0, 0, 0, 1, 1])
    cm = compute_confusion(y_true, y_pred, labels=[0, 1])
    assert cm.shape == (2, 2)
    assert cm[1, 0] == 1  # true 1 pred 0
    assert cm[0, 1] == 1  # true 0 pred 1

    errs = top_misclassified(
        texts, y_true, y_pred, target_names=["neg", "pos"], top_k=5
    )
    assert len(errs) == 2
    assert errs[0]["true_name"] in {"neg", "pos"}
    assert "text" in errs[0]


def test_compute_metrics_keys():
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 1])
    m = compute_metrics(y_true, y_pred, target_names=["a", "b"])
    assert "accuracy" in m and "macro_f1" in m and "report" in m
    assert 0.0 <= m["accuracy"] <= 1.0
