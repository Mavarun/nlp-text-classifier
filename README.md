# nlp-text-classifier

Research slice: **leakage-safe text classification** comparing a **TF-IDF + LogisticRegression** floor to a **tiny from-scratch TransformerEncoder** (no pretrained HF weights) on a frozen stratified 20 Newsgroups subset.

Public data only (`sklearn.datasets.fetch_20newsgroups`). Methodology demo — **not** a production accuracy claim.

## Today's hypothesis

1. On a frozen stratified train/test split of a public text set (20 Newsgroups subset), a TF-IDF + LogisticRegression baseline sets a strong leakage-safe floor.
2. A small from-scratch TransformerEncoder classifier (no pretrained HF weights; bag-of-token embeddings + encoder) trained only on the train split can beat or match that floor on macro-F1 when class semantics are separable.
3. Confusion-matrix + top misclassified examples will show where bag-of-words confuses topic neighbors vs where the tiny transformer helps or fails.

## Method

- **Data**: 4 categories (`alt.atheism`, `talk.religion.misc`, `comp.graphics`, `sci.space`), headers/footers/quotes removed, stratified subsample to 1200 docs, then `train_test_split(..., test_size=0.25, stratify=y, random_state=42)` → 900 train / 300 test.
- **Leakage wall**: TF-IDF vocabulary and transformer vocab are fit on **train texts only**.
- **Baseline**: `TfidfVectorizer(min_df=2, ngram_range=(1,2), max_features=20000)` → `LogisticRegression(max_iter=1000)`.
- **Transformer**: whitespace/alnum tokenize → train-only vocab (`min_freq=2`) → `Embedding(d=64)` + learned positional → `nn.TransformerEncoder` (2 layers, `nhead=4`) → mean-pool → linear head. Train 6 epochs, `max_len=128`, Adam `lr=1e-3`, seed 42. **No pretrained weights.**
- **Metrics**: accuracy, macro-F1, per-class report; confusion matrix + top misclassified snippets.

## How to run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
pytest
python scripts/run_newsgroups_slice.py
```

First run may download the 20 Newsgroups corpus via scikit-learn (unit tests use synthetic text and do not require the download).

## Results (this slice, frozen split seed=42)

| Model | accuracy | macro-F1 |
| --- | ---: | ---: |
| TF-IDF + LogisticRegression | **0.727** | **0.700** |
| Tiny TransformerEncoder (from scratch) | 0.493 | 0.473 |

**Hypothesis (1)** holds: the TF-IDF + LR floor is strong on this separable-ish topic mix.

**Hypothesis (2)** did **not** hold on this CPU-tiny setup: the from-scratch encoder lagged the bag-of-ngrams floor by ~0.23 macro-F1. With only ~900 train docs, `d_model=64`, 2 layers, and 6 epochs, the model underfits relative to a well-tuned sparse linear baseline. That is an honest negative result for “tiny random-init transformer beats TF-IDF” *here*, not a claim that transformers fail in general.

**Hypothesis (3)** is visible in the matrices: both models confuse `alt.atheism` ↔ `talk.religion.misc` (topic neighbors). Baseline still recovers religion better via sparse n-grams; the tiny transformer collapses more of those neighbor errors and also leaks graphics↔space tokens under short `max_len=128` truncation.

## Assumptions and limits

- 4-class subset + 1200-doc cap for speed; not the full 20 Newsgroups benchmark.
- Headers/footers/quotes stripped — remaining signature lines / quoted fragments can still leak cues.
- Tiny transformer has no pretrained embeddings, no BPE, short context, few epochs — expect TF-IDF to win until capacity/data increase.
- Macro-F1 on 300 test docs is noisy; do not treat deltas of a few points as stable.
- No hyperparameter search, no pretrained HF models, **no production accuracy claim**.

## Layout

```
src/nlp_text/
  data.py             # 20newsgroups subset + frozen stratified split
  baseline.py         # TF-IDF + LogisticRegression
  transformer_clf.py  # tiny torch TransformerEncoder classifier
  evaluate.py         # accuracy / macro-F1 / report
  errors.py           # confusion matrix + top misclassified
scripts/run_newsgroups_slice.py
tests/
```

## Next slices (not done)

- Longer training / larger `d_model` or pretrained token embeddings (still train-only fine-tune)
- DistilBERT / MiniLM **with** frozen split + honest comparison to this TF-IDF floor
- Calibration / ECE on text probabilities
- Error slices focused on atheism↔religion neighbor confusion
