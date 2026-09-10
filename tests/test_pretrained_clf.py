"""Frozen MiniLM-style embeddings + LR: leakage, shapes, seed determinism."""

import numpy as np

from nlp_text.data import frozen_stratified_split
from nlp_text.pretrained_clf import (
    fit_pretrained_logreg,
    predict_pretrained_logreg,
)
from nlp_text.pretrained_encoder import build_pretrained_encoder


def _separable_corpus():
    texts, y = [], []
    for c, word in enumerate(["alpha", "beta", "gamma", "delta"]):
        for i in range(12):
            texts.append(f"{word} {word} example {i} {word} marker")
            y.append(c)
    return texts, np.array(y, dtype=np.int64)


def test_pretrained_clf_shapes_and_labels():
    texts, y = _separable_corpus()
    split = frozen_stratified_split(texts, y, test_size=0.25, random_state=42)
    enc = build_pretrained_encoder(force_offline_hash=True)
    model = fit_pretrained_logreg(
        split.X_train, split.y_train, encoder=enc, random_state=42
    )
    preds = predict_pretrained_logreg(model, split.X_test)
    assert preds.shape == split.y_test.shape
    assert set(preds.tolist()).issubset(set(range(4)))


def test_no_test_leakage_in_linear_head():
    """Linear head is fit on train embeddings only; test texts never enter clf.fit."""
    train = ["cat dog bird", "cat fish bird", "dog bird cat", "fish cat dog"]
    y_train = np.array([0, 0, 1, 1])
    test = ["zzzzuniqueonlyintest aaa", "zzzzuniqueonlyintest bbb"]
    poison = "zzzzuniqueonlyintest"

    fit_batches: list[list[str]] = []

    def tracking_encode(texts):
        batch = list(texts)
        # Record only non-probe batches that look like train/test payloads
        if batch != ["probe"]:
            fit_batches.append(batch)
        out = np.zeros((len(texts), 16), dtype=np.float32)
        for i, t in enumerate(texts):
            out[i, hash(t) % 16] = 1.0
            out[i, (hash(t) // 16) % 16] = 0.5
        return out

    enc = build_pretrained_encoder(encode_fn=tracking_encode)
    fit_batches.clear()  # drop dim-probe if any slipped through before clear
    model = fit_pretrained_logreg(train, y_train, encoder=enc, random_state=0)
    # Fit should encode train only (no poison token)
    assert len(fit_batches) == 1
    assert fit_batches[0] == train
    assert all(poison not in t for t in fit_batches[0])

    coef_before = model.clf.coef_.copy()
    _ = predict_pretrained_logreg(model, test)
    # Predict may encode test, but coefficients stay fixed (no refit)
    np.testing.assert_array_equal(model.clf.coef_, coef_before)
    assert poison not in " ".join(fit_batches[0])


def test_deterministic_seed_same_predictions():
    texts, y = _separable_corpus()
    split = frozen_stratified_split(texts, y, test_size=0.25, random_state=42)
    enc = build_pretrained_encoder(force_offline_hash=True)
    m1 = fit_pretrained_logreg(split.X_train, split.y_train, encoder=enc, random_state=42)
    m2 = fit_pretrained_logreg(split.X_train, split.y_train, encoder=enc, random_state=42)
    p1 = predict_pretrained_logreg(m1, split.X_test)
    p2 = predict_pretrained_logreg(m2, split.X_test)
    np.testing.assert_array_equal(p1, p2)
