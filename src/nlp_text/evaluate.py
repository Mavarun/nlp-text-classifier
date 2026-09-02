"""Accuracy, macro-F1, and per-class classification report helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    target_names: list[str] | None = None,
) -> dict[str, Any]:
    """Return accuracy, macro-F1, and sklearn classification_report dict."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    report = classification_report(
        y_true,
        y_pred,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "report": report,
    }


def format_metrics(name: str, metrics: dict[str, Any]) -> str:
    """Human-readable one-liner + report block."""
    lines = [
        f"=== {name} ===",
        f"accuracy={metrics['accuracy']:.4f}  macro_f1={metrics['macro_f1']:.4f}",
        classification_report_from_dict(metrics["report"]),
    ]
    return "\n".join(lines)


def classification_report_from_dict(report: dict[str, Any]) -> str:
    """Compact text table from sklearn report dict (for logging)."""
    rows = []
    for key, val in report.items():
        if not isinstance(val, dict):
            continue
        rows.append(
            f"  {key:28s} p={val.get('precision', 0):.3f} "
            f"r={val.get('recall', 0):.3f} f1={val.get('f1-score', 0):.3f} "
            f"n={int(val.get('support', 0))}"
        )
    return "\n".join(rows)
