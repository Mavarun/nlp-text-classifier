#!/usr/bin/env python3
"""Run frozen MiniLM embeddings + LR vs TF-IDF on the same 20 Newsgroups split.

Offline: NLP_TEXT_FORCE_HASH=1 uses deterministic hash embeddings (not for
metrics claims). Model cache: set SENTENCE_TRANSFORMERS_HOME or pass via
sentence-transformers defaults (~/.cache/huggingface).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nlp_text.baseline import fit_baseline, predict_baseline
from nlp_text.errors import compute_confusion, top_misclassified
from nlp_text.evaluate import compute_metrics, format_metrics
from nlp_text.pretrained_clf import fit_pretrained_logreg, predict_pretrained_logreg
from nlp_text.pretrained_encoder import build_pretrained_encoder
from nlp_text.pretrained_split import (
    PRETRAINED_CATEGORIES,
    load_pretrained_slice_split,
)


def _neighbor_confusion(cm, names: list[str]) -> dict:
    """Count atheism↔religion bidirectional confusions when both labels exist."""
    try:
        i_a = names.index("alt.atheism")
        i_r = names.index("talk.religion.misc")
    except ValueError:
        return {}
    return {
        "atheism_pred_religion": int(cm[i_a, i_r]),
        "religion_pred_atheism": int(cm[i_r, i_a]),
        "atheism_religion_total": int(cm[i_a, i_r] + cm[i_r, i_a]),
    }


def main() -> None:
    force_hash = os.environ.get("NLP_TEXT_FORCE_HASH", "").lower() in {"1", "true", "yes"}
    print("Categories:", list(PRETRAINED_CATEGORIES))
    split = load_pretrained_slice_split()
    print(
        f"n_train={len(split.X_train)} n_test={len(split.X_test)} "
        f"n_classes={len(split.target_names)}"
    )
    print("target_names:", split.target_names)

    # --- TF-IDF baseline (same split) ---
    baseline = fit_baseline(split.X_train, split.y_train)
    baseline_pred = predict_baseline(baseline, split.X_test)
    baseline_metrics = compute_metrics(
        split.y_test, baseline_pred, target_names=split.target_names
    )
    print(format_metrics("TF-IDF + LogisticRegression", baseline_metrics))

    # --- Frozen MiniLM + LR ---
    enc = build_pretrained_encoder(force_offline_hash=force_hash)
    print(f"encoder={enc.model_name} dim={enc.dim} offline={enc.offline}")
    pre = fit_pretrained_logreg(
        split.X_train, split.y_train, encoder=enc, random_state=42
    )
    pre_pred = predict_pretrained_logreg(pre, split.X_test)
    pre_metrics = compute_metrics(
        split.y_test, pre_pred, target_names=split.target_names
    )
    print(format_metrics("MiniLM (frozen) + LogisticRegression", pre_metrics))

    labels = list(range(len(split.target_names)))
    cm_base = compute_confusion(split.y_test, baseline_pred, labels=labels)
    cm_pre = compute_confusion(split.y_test, pre_pred, labels=labels)
    print("TF-IDF confusion matrix (rows=true, cols=pred):")
    print(cm_base)
    print("MiniLM confusion matrix (rows=true, cols=pred):")
    print(cm_pre)

    nb_base = _neighbor_confusion(cm_base, split.target_names)
    nb_pre = _neighbor_confusion(cm_pre, split.target_names)
    print("Neighbor confusions (atheism↔religion):")
    print("  TF-IDF:", nb_base)
    print("  MiniLM:", nb_pre)

    print("MiniLM top misclassified:")
    for row in top_misclassified(
        split.X_test,
        split.y_test,
        pre_pred,
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
        "encoder": enc.model_name,
        "encoder_offline": enc.offline,
        "baseline": {
            "accuracy": baseline_metrics["accuracy"],
            "macro_f1": baseline_metrics["macro_f1"],
            "atheism_religion_confusions": nb_base,
        },
        "minilm_logreg": {
            "accuracy": pre_metrics["accuracy"],
            "macro_f1": pre_metrics["macro_f1"],
            "atheism_religion_confusions": nb_pre,
        },
    }
    out = ROOT / "artifacts"
    out.mkdir(exist_ok=True)
    path = out / "pretrained_metrics.json"
    path.write_text(json.dumps(summary, indent=2))
    print(f"Wrote {path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
