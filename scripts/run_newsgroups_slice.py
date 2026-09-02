#!/usr/bin/env python3
"""Run TF-IDF baseline vs tiny TransformerEncoder on frozen 20 Newsgroups split."""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Allow running without install: add src to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nlp_text.baseline import fit_baseline, predict_baseline
from nlp_text.data import DEFAULT_CATEGORIES, load_frozen_newsgroups_split
from nlp_text.errors import compute_confusion, top_misclassified
from nlp_text.evaluate import compute_metrics, format_metrics
from nlp_text.transformer_clf import predict_transformer, set_seed, train_transformer


def main() -> None:
    set_seed(42)
    print("Categories:", list(DEFAULT_CATEGORIES))
    split = load_frozen_newsgroups_split(max_docs=1200)
    print(
        f"n_train={len(split.X_train)} n_test={len(split.X_test)} "
        f"n_classes={len(split.target_names)}"
    )
    print("target_names:", split.target_names)

    # --- Baseline ---
    baseline = fit_baseline(split.X_train, split.y_train)
    baseline_pred = predict_baseline(baseline, split.X_test)
    baseline_metrics = compute_metrics(
        split.y_test, baseline_pred, target_names=split.target_names
    )
    print(format_metrics("TF-IDF + LogisticRegression", baseline_metrics))

    # --- Tiny transformer ---
    trained = train_transformer(
        split.X_train,
        split.y_train,
        n_classes=len(split.target_names),
        epochs=6,
        batch_size=32,
        lr=1e-3,
        max_len=128,
        min_freq=2,
        seed=42,
    )
    tr_pred = predict_transformer(trained, split.X_test)
    tr_metrics = compute_metrics(
        split.y_test, tr_pred, target_names=split.target_names
    )
    print(format_metrics("Tiny TransformerEncoder", tr_metrics))

    # --- Error analysis (baseline) ---
    cm = compute_confusion(
        split.y_test, baseline_pred, labels=list(range(len(split.target_names)))
    )
    print("Baseline confusion matrix (rows=true, cols=pred):")
    print(cm)
    print("Baseline top misclassified:")
    for row in top_misclassified(
        split.X_test,
        split.y_test,
        baseline_pred,
        target_names=split.target_names,
        top_k=5,
    ):
        print(
            f"  [{row['index']}] true={row['true_name']} pred={row['pred_name']}: "
            f"{row['text'][:120]}"
        )

    cm_tr = compute_confusion(
        split.y_test, tr_pred, labels=list(range(len(split.target_names)))
    )
    print("Transformer confusion matrix (rows=true, cols=pred):")
    print(cm_tr)
    print("Transformer top misclassified:")
    for row in top_misclassified(
        split.X_test,
        split.y_test,
        tr_pred,
        target_names=split.target_names,
        top_k=5,
    ):
        print(
            f"  [{row['index']}] true={row['true_name']} pred={row['pred_name']}: "
            f"{row['text'][:120]}"
        )

    summary = {
        "n_train": len(split.X_train),
        "n_test": len(split.X_test),
        "categories": split.target_names,
        "baseline": {
            "accuracy": baseline_metrics["accuracy"],
            "macro_f1": baseline_metrics["macro_f1"],
        },
        "transformer": {
            "accuracy": tr_metrics["accuracy"],
            "macro_f1": tr_metrics["macro_f1"],
        },
    }
    out = ROOT / "artifacts"
    out.mkdir(exist_ok=True)
    path = out / "metrics.json"
    path.write_text(json.dumps(summary, indent=2))
    print(f"Wrote {path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
