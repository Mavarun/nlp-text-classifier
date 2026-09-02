"""Load a 20 Newsgroups subset and build a frozen stratified train/test split."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split

# Related + distinct topics; small enough for CPU training in minutes.
DEFAULT_CATEGORIES: tuple[str, ...] = (
    "alt.atheism",
    "talk.religion.misc",
    "comp.graphics",
    "sci.space",
)

RANDOM_STATE = 42
TEST_SIZE = 0.25


@dataclass(frozen=True)
class TextSplit:
    """Frozen stratified text classification split."""

    X_train: list[str]
    X_test: list[str]
    y_train: np.ndarray
    y_test: np.ndarray
    target_names: list[str]
    train_indices: np.ndarray
    test_indices: np.ndarray


def load_newsgroups_subset(
    categories: Sequence[str] | None = None,
    *,
    remove: tuple[str, ...] = ("headers", "footers", "quotes"),
    subset: str = "all",
) -> tuple[list[str], np.ndarray, list[str]]:
    """Fetch a public 20 Newsgroups subset (may download once via sklearn)."""
    cats = list(categories) if categories is not None else list(DEFAULT_CATEGORIES)
    bunch = fetch_20newsgroups(
        subset=subset,
        categories=cats,
        remove=remove,
        shuffle=False,
        random_state=RANDOM_STATE,
    )
    texts = list(bunch.data)
    y = np.asarray(bunch.target, dtype=np.int64)
    return texts, y, list(bunch.target_names)


def frozen_stratified_split(
    texts: Sequence[str],
    y: np.ndarray,
    *,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
    target_names: Sequence[str] | None = None,
) -> TextSplit:
    """Stratified train/test split with fixed seed (leakage-safe floor).

    Indices are regenerated deterministically from ``random_state``; callers
    should fit vectorizers / vocabs on train only.
    """
    texts_arr = np.asarray(list(texts), dtype=object)
    y = np.asarray(y, dtype=np.int64)
    idx = np.arange(len(texts_arr))
    train_idx, test_idx = train_test_split(
        idx,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    train_idx = np.sort(train_idx)
    test_idx = np.sort(test_idx)
    names = list(target_names) if target_names is not None else [
        str(i) for i in range(int(y.max()) + 1)
    ]
    return TextSplit(
        X_train=texts_arr[train_idx].tolist(),
        X_test=texts_arr[test_idx].tolist(),
        y_train=y[train_idx],
        y_test=y[test_idx],
        target_names=names,
        train_indices=train_idx,
        test_indices=test_idx,
    )


def load_frozen_newsgroups_split(
    categories: Sequence[str] | None = None,
    *,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
    max_docs: int | None = 1200,
) -> TextSplit:
    """Convenience: load subset, optionally subsample, then frozen stratified split."""
    texts, y, names = load_newsgroups_subset(categories)
    if max_docs is not None and len(texts) > max_docs:
        # Deterministic stratified subsample before the frozen train/test split.
        idx = np.arange(len(texts))
        keep, _ = train_test_split(
            idx,
            train_size=max_docs,
            random_state=random_state,
            stratify=y,
        )
        keep = np.sort(keep)
        texts = [texts[i] for i in keep]
        y = y[keep]
    return frozen_stratified_split(
        texts, y, test_size=test_size, random_state=random_state, target_names=names
    )
