"""Pretrained slice must reuse the identical frozen stratified split."""

import numpy as np

from nlp_text.data import frozen_stratified_split, load_frozen_newsgroups_split
from nlp_text.pretrained_split import (
    PRETRAINED_MAX_DOCS,
    PRETRAINED_RANDOM_STATE,
    assert_same_frozen_indices,
    load_pretrained_slice_split,
)


def _synthetic(n_per_class: int = 20, n_classes: int = 4):
    texts, y = [], []
    for c in range(n_classes):
        for i in range(n_per_class):
            texts.append(f"class{c} token{c} doc{i} filler")
            y.append(c)
    return texts, np.array(y, dtype=np.int64)


def test_pretrained_split_matches_frozen_helper():
    texts, y = _synthetic()
    a = frozen_stratified_split(texts, y, random_state=PRETRAINED_RANDOM_STATE)
    b = frozen_stratified_split(texts, y, random_state=PRETRAINED_RANDOM_STATE)
    assert_same_frozen_indices(a, b)
    np.testing.assert_array_equal(a.train_indices, b.train_indices)


def test_pretrained_constants_align_with_first_slice():
    assert PRETRAINED_RANDOM_STATE == 42
    assert PRETRAINED_MAX_DOCS == 1200
