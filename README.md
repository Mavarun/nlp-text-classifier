# nlp-text-classifier

Research slices: **leakage-safe text classification** on a frozen stratified 20 Newsgroups subset.

1. **Slice A**: TF-IDF + LogisticRegression vs a **tiny from-scratch TransformerEncoder** (no pretrained HF weights).
2. **Slice B (this PR)**: same frozen split — **frozen MiniLM sentence embeddings + LogisticRegression** vs the TF-IDF floor.

Public data only (`sklearn.datasets.fetch_20newsgroups`). Methodology demo — **not** a production accuracy claim.

## Slice B hypothesis (pretrained MiniLM baseline)

1. On the same frozen 20 Newsgroups split (seed 42), a small pretrained sentence encoder (MiniLM or DistilBERT via sentence-transformers / transformers) should beat TF-IDF+LR on macro-F1.
2. Error analysis should show fewer atheism↔religion confusions than bag-of-words if semantics help.
3. Keep leakage rules: vocab/encoder fit only on train; report OOS only; not a production claim.

## Method (both slices)

- **Data**: 4 categories (`alt.atheism`, `talk.religion.misc`, `comp.graphics`, `sci.space`), headers/footers/quotes removed, stratified subsample to 1200 docs, then `train_test_split(..., test_size=0.25, stratify=y, random_state=42)` → 900 train / 300 test.
- **Leakage wall**: TF-IDF vocabulary / transformer vocab / logistic head are fit on **train texts only**. MiniLM weights are external pretrained and **frozen** (not fine-tuned on this split); train embeddings only enter `LogisticRegression.fit`.
- **Baseline**: `TfidfVectorizer(min_df=2, ngram_range=(1,2), max_features=20000)` → `LogisticRegression(max_iter=1000)`.
- **Tiny transformer (slice A)**: whitespace/alnum tokenize → train-only vocab → `Embedding(d=64)` + `nn.TransformerEncoder` (2 layers). **No pretrained weights.**
- **MiniLM (slice B)**: `sentence-transformers/all-MiniLM-L6-v2` (384-d, normalized) → frozen encode → `LogisticRegression` head. Prefer frozen embeddings + LR for CPU speed/reliability.
- **Metrics**: accuracy, macro-F1, per-class report; confusion matrix + atheism↔religion neighbor counts.

### Offline / cache notes

- First MiniLM run downloads weights into the Hugging Face / sentence-transformers cache (`HF_HOME` / `SENTENCE_TRANSFORMERS_HOME`).
- Unit tests inject a fake `encode_fn` or use `NLP_TEXT_FORCE_HASH=1` (deterministic hash embeddings) — **no network**, not for metrics claims.
- If download fails: `NLP_TEXT_ALLOW_HASH_FALLBACK=1` enables the hash path for dry-runs.

## How to run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
pytest
python scripts/run_newsgroups_slice.py      # slice A: TF-IDF vs tiny transformer
python scripts/run_pretrained_slice.py      # slice B: TF-IDF vs frozen MiniLM + LR
```

First run may download the 20 Newsgroups corpus via scikit-learn and (for slice B) MiniLM weights. Unit tests use synthetic text / hash embeddings and do not require those downloads.

## Results — Slice A (frozen split seed=42)

| Model | accuracy | macro-F1 |
| --- | ---: | ---: |
| TF-IDF + LogisticRegression | **0.727** | **0.700** |
| Tiny TransformerEncoder (from scratch) | 0.493 | 0.473 |

TF-IDF floor held; tiny random-init transformer underfit on ~900 docs.

## Results — Slice B (same frozen split seed=42)

| Model | accuracy | macro-F1 | atheism↔religion confusions |
| --- | ---: | ---: | ---: |
| TF-IDF + LogisticRegression | 0.727 | 0.700 | **18** |
| MiniLM (frozen) + LogisticRegression | **0.737** | **0.713** | 37 |

**Hypothesis (1)** holds narrowly: frozen MiniLM + LR beats TF-IDF by ~0.013 macro-F1 (0.713 vs 0.700) on the same 300 OOS docs.

**Hypothesis (2)** did **not** hold: atheism↔religion bidirectional confusions **increased** under MiniLM (37 vs 18). Semantic embeddings help `comp.graphics` / `sci.space` separation, but the atheism/religion neighbor pair remains (and worsens) — short Usenet rant style may sit close in MiniLM space while sparse religion n-grams still give TF-IDF an edge on that pair.

**Hypothesis (3)** leakage rules held: encoder frozen (external), LR fit on train embeddings only; metrics reported on test only. Not a production claim.

## Assumptions and limits

- 4-class subset + 1200-doc cap for speed; not the full 20 Newsgroups benchmark.
- Macro-F1 on 300 test docs is noisy; a +0.013 delta is directionally interesting, not definitive.
- Frozen MiniLM is not task-adapted; light fine-tuning might change the neighbor-error story (CPU cost).
- Headers/footers/quotes stripped — residual cues can remain.
- **No production accuracy claim.**

## Layout

```
src/nlp_text/
  data.py                 # 20newsgroups subset + frozen stratified split
  baseline.py             # TF-IDF + LogisticRegression
  transformer_clf.py      # tiny torch TransformerEncoder classifier
  pretrained_split.py     # reuses identical frozen split for MiniLM slice
  pretrained_encoder.py   # MiniLM wrapper + offline hash fallback
  pretrained_clf.py       # frozen embeddings + LR head
  evaluate.py             # accuracy / macro-F1 / report
  errors.py               # confusion matrix + top misclassified
scripts/run_newsgroups_slice.py
scripts/run_pretrained_slice.py
tests/
```

## Next slices (not done)

- Light MiniLM fine-tune (still train-only) vs frozen embeddings
- Calibration / ECE on text probabilities
- Focused error slices / hard-negative mining on atheism↔religion
