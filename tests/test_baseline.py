"""Baseline TF-IDF + LR tests; leakage check that vectorizer is train-only."""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from nlp_text.baseline import build_tfidf_logreg, fit_baseline, predict_baseline
from nlp_text.data import frozen_stratified_split


def _corpus():
    texts = []
    y = []
    for c, word in enumerate(["alpha", "beta", "gamma", "delta"]):
        for i in range(15):
            texts.append(f"{word} {word} example {i} unique_{word}_{i}")
            y.append(c)
    return texts, np.array(y, dtype=np.int64)


def test_baseline_fits_and_predicts():
    texts, y = _corpus()
    split = frozen_stratified_split(texts, y, test_size=0.25, random_state=42)
    model = fit_baseline(split.X_train, split.y_train, min_df=1, max_features=5000)
    preds = predict_baseline(model, split.X_test)
    assert preds.shape == split.y_test.shape
    assert set(preds.tolist()).issubset(set(range(4)))


def test_vectorizer_not_fit_on_test_vocab_leakage():
    """Train-only fit: a token that appears only in test must be absent from vocab."""
    train_texts = ["cat dog bird", "cat fish bird", "dog bird cat"]
    test_texts = ["zzzzuniqueonlyintest aaa", "zzzzuniqueonlyintest bbb"]
    y_train = np.array([0, 0, 1])

    pipe = build_tfidf_logreg(min_df=1, max_features=1000)
    pipe.fit(train_texts, y_train)
    vocab = pipe.named_steps["tfidf"].vocabulary_
    assert "zzzzuniqueonlyintest" not in vocab

    # Transforming test still works (unknown tokens ignored), without expanding vocab
    X_test = pipe.named_steps["tfidf"].transform(test_texts)
    assert X_test.shape[0] == 2
    assert "zzzzuniqueonlyintest" not in pipe.named_steps["tfidf"].vocabulary_


def test_refit_does_not_see_held_out_tokens():
    texts, y = _corpus()
    # Inject a token only into what will become test after stratified split
    texts, y = list(texts), y.copy()
    # Use frozen split then ensure a made-up test-only token isn't in train vocab
    split = frozen_stratified_split(texts, y, test_size=0.25, random_state=42)
    poison = "supersecretleakagetokenxyz"
    split_X_test = [t + " " + poison for t in split.X_test]
    model = fit_baseline(split.X_train, split.y_train, min_df=1)
    vocab = model.named_steps["tfidf"].vocabulary_
    assert poison not in vocab
    _ = predict_baseline(model, split_X_test)
    assert poison not in model.named_steps["tfidf"].vocabulary_
