"""Unit tests for frozen stratified split helpers (synthetic, no download)."""

import numpy as np

from nlp_text.data import frozen_stratified_split


def _synthetic_corpus(n_per_class: int = 20, n_classes: int = 4):
    texts = []
    y = []
    for c in range(n_classes):
        for i in range(n_per_class):
            texts.append(f"class{c} token{c} doc{i} shared filler words here")
            y.append(c)
    return texts, np.array(y, dtype=np.int64)


def test_frozen_split_sizes_and_stratification():
    texts, y = _synthetic_corpus()
    split = frozen_stratified_split(texts, y, test_size=0.25, random_state=42)
    assert len(split.X_train) + len(split.X_test) == len(texts)
    assert len(split.y_train) == len(split.X_train)
    assert len(split.y_test) == len(split.X_test)
    # Stratified: each class appears in both splits
    assert set(split.y_train.tolist()) == set(range(4))
    assert set(split.y_test.tolist()) == set(range(4))


def test_frozen_split_deterministic():
    texts, y = _synthetic_corpus()
    a = frozen_stratified_split(texts, y, test_size=0.25, random_state=42)
    b = frozen_stratified_split(texts, y, test_size=0.25, random_state=42)
    np.testing.assert_array_equal(a.train_indices, b.train_indices)
    np.testing.assert_array_equal(a.test_indices, b.test_indices)
    assert a.X_train == b.X_train


def test_train_test_indices_disjoint():
    texts, y = _synthetic_corpus()
    split = frozen_stratified_split(texts, y, random_state=42)
    assert len(set(split.train_indices) & set(split.test_indices)) == 0
