"""Tiny transformer unit tests on synthetic data (no network)."""

import numpy as np

from nlp_text.transformer_clf import (
    build_vocab,
    predict_transformer,
    simple_tokenize,
    train_transformer,
)


def test_simple_tokenize():
    assert simple_tokenize("Hello, WORLD! 123") == ["hello", "world", "123"]


def test_vocab_from_train_only():
    train = ["red red car", "blue blue car", "red truck"]
    vocab = build_vocab(train, min_freq=1)
    assert "red" in vocab.stoi
    assert "zzzzmissing" not in vocab.stoi
    assert vocab.pad_id == 0
    assert vocab.unk_id == 1


def test_transformer_overfits_tiny_separable_set():
    """With repeated class keywords, a tiny encoder should nearly memorize train."""
    texts = []
    y = []
    for c, word in enumerate(["alpha", "bravo", "charlie", "delta"]):
        for i in range(8):
            texts.append(f"{word} {word} sample {i} {word}")
            y.append(c)
    y = np.array(y, dtype=np.int64)
    trained = train_transformer(
        texts,
        y,
        n_classes=4,
        epochs=12,
        batch_size=8,
        lr=3e-3,
        max_len=32,
        min_freq=1,
        seed=0,
    )
    preds = predict_transformer(trained, texts)
    acc = (preds == y).mean()
    assert acc >= 0.9, f"expected near-perfect train fit, got acc={acc}"
