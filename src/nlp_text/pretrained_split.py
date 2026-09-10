"""Reuse the frozen 20 Newsgroups split for the pretrained MiniLM slice.

This module intentionally wraps ``load_frozen_newsgroups_split`` so the
pretrained baseline compares on the **identical** seed-42 stratified split
as the TF-IDF / tiny-transformer slice (leakage-safe, train-only fit).
"""

from __future__ import annotations

from typing import Sequence

from nlp_text.data import (
    DEFAULT_CATEGORIES,
    RANDOM_STATE,
    TEST_SIZE,
    TextSplit,
    frozen_stratified_split,
    load_frozen_newsgroups_split,
)

# Re-export for callers / tests that pin the pretrained slice contract.
PRETRAINED_CATEGORIES = DEFAULT_CATEGORIES
PRETRAINED_RANDOM_STATE = RANDOM_STATE
PRETRAINED_TEST_SIZE = TEST_SIZE
PRETRAINED_MAX_DOCS = 1200


def load_pretrained_slice_split(
    categories: Sequence[str] | None = None,
    *,
    test_size: float = PRETRAINED_TEST_SIZE,
    random_state: int = PRETRAINED_RANDOM_STATE,
    max_docs: int | None = PRETRAINED_MAX_DOCS,
) -> TextSplit:
    """Return the same frozen stratified split used by the first research slice.

    Vocab / encoder fitting must still happen on ``split.X_train`` only;
    metrics are reported on ``split.X_test`` (OOS) only.
    """
    return load_frozen_newsgroups_split(
        categories,
        test_size=test_size,
        random_state=random_state,
        max_docs=max_docs,
    )


def assert_same_frozen_indices(
    a: TextSplit,
    b: TextSplit,
) -> None:
    """Raise AssertionError if two splits do not share train/test indices."""
    import numpy as np

    if not np.array_equal(a.train_indices, b.train_indices):
        raise AssertionError("train_indices diverge from frozen split")
    if not np.array_equal(a.test_indices, b.test_indices):
        raise AssertionError("test_indices diverge from frozen split")
