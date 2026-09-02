"""Tiny from-scratch TransformerEncoder text classifier (no pretrained HF weights)."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
TOKEN_RE = re.compile(r"[A-Za-z0-9']+")


def set_seed(seed: int = 42) -> None:
    """Seed Python / NumPy / Torch for reproducibility."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def simple_tokenize(text: str) -> list[str]:
    """Lowercase whitespace/punctuation-aware tokenizer (bag-of-token style)."""
    return TOKEN_RE.findall(text.lower())


@dataclass
class Vocab:
    """Train-only vocabulary with pad/unk."""

    stoi: dict[str, int]
    itos: list[str]

    @property
    def pad_id(self) -> int:
        return self.stoi[PAD_TOKEN]

    @property
    def unk_id(self) -> int:
        return self.stoi[UNK_TOKEN]

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, tokens: Sequence[str], max_len: int) -> list[int]:
        ids = [self.stoi.get(t, self.unk_id) for t in tokens[:max_len]]
        if len(ids) < max_len:
            ids = ids + [self.pad_id] * (max_len - len(ids))
        return ids


def build_vocab(
    texts: Sequence[str],
    *,
    min_freq: int = 2,
    max_size: int = 20_000,
) -> Vocab:
    """Build vocab from train texts only (leakage-safe)."""
    counts: Counter[str] = Counter()
    for text in texts:
        counts.update(simple_tokenize(text))
    itos = [PAD_TOKEN, UNK_TOKEN]
    for tok, freq in counts.most_common():
        if freq < min_freq:
            continue
        if tok in (PAD_TOKEN, UNK_TOKEN):
            continue
        itos.append(tok)
        if len(itos) >= max_size:
            break
    stoi = {t: i for i, t in enumerate(itos)}
    return Vocab(stoi=stoi, itos=itos)


class TextDataset(Dataset):
    def __init__(
        self,
        texts: Sequence[str],
        labels: np.ndarray,
        vocab: Vocab,
        max_len: int = 128,
    ) -> None:
        self.texts = list(texts)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = simple_tokenize(self.texts[idx])
        ids = self.vocab.encode(tokens, self.max_len)
        x = torch.tensor(ids, dtype=torch.long)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y


class TinyTransformerClassifier(nn.Module):
    """Embedding + positional + TransformerEncoder + mean-pool + linear head."""

    def __init__(
        self,
        vocab_size: int,
        n_classes: int,
        *,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        max_len: int = 128,
        dropout: float = 0.1,
        pad_id: int = 0,
    ) -> None:
        super().__init__()
        self.pad_id = pad_id
        self.max_len = max_len
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_embedding = nn.Embedding(max_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # token_ids: (B, L)
        bsz, seqlen = token_ids.shape
        positions = torch.arange(seqlen, device=token_ids.device).unsqueeze(0).expand(bsz, -1)
        mask = token_ids.eq(self.pad_id)  # True = pad (ignored by encoder)
        x = self.embedding(token_ids) * math.sqrt(self.d_model)
        x = x + self.pos_embedding(positions)
        x = self.encoder(x, src_key_padding_mask=mask)
        # Mean-pool over non-pad tokens
        valid = (~mask).unsqueeze(-1).float()  # (B, L, 1)
        summed = (x * valid).sum(dim=1)
        denom = valid.sum(dim=1).clamp(min=1.0)
        pooled = summed / denom
        pooled = self.dropout(pooled)
        return self.head(pooled)


@dataclass
class TrainedTransformer:
    """Bundle of trained model + vocab + metadata for prediction."""

    model: TinyTransformerClassifier
    vocab: Vocab
    max_len: int
    device: torch.device
    target_names: list[str] | None = None


def train_transformer(
    texts: Sequence[str],
    y: np.ndarray,
    *,
    n_classes: int | None = None,
    epochs: int = 5,
    batch_size: int = 32,
    lr: float = 1e-3,
    max_len: int = 128,
    min_freq: int = 2,
    d_model: int = 64,
    nhead: int = 4,
    num_layers: int = 2,
    seed: int = 42,
    device: str | None = None,
) -> TrainedTransformer:
    """Fit vocab on train texts only, then train tiny TransformerEncoder."""
    set_seed(seed)
    y = np.asarray(y, dtype=np.int64)
    if n_classes is None:
        n_classes = int(y.max()) + 1
    vocab = build_vocab(texts, min_freq=min_freq)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = TinyTransformerClassifier(
        vocab_size=len(vocab),
        n_classes=n_classes,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        max_len=max_len,
        pad_id=vocab.pad_id,
    ).to(dev)

    ds = TextDataset(texts, y, vocab, max_len=max_len)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb = xb.to(dev)
            yb = yb.to(dev)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            opt.step()

    return TrainedTransformer(model=model, vocab=vocab, max_len=max_len, device=dev)


@torch.no_grad()
def predict_transformer(trained: TrainedTransformer, texts: Sequence[str]) -> np.ndarray:
    """Predict class labels with a trained tiny transformer."""
    model = trained.model
    model.eval()
    ds = TextDataset(texts, np.zeros(len(texts), dtype=np.int64), trained.vocab, trained.max_len)
    loader = DataLoader(ds, batch_size=64, shuffle=False)
    preds: list[np.ndarray] = []
    for xb, _ in loader:
        xb = xb.to(trained.device)
        logits = model(xb)
        preds.append(logits.argmax(dim=-1).cpu().numpy())
    if not preds:
        return np.array([], dtype=np.int64)
    return np.concatenate(preds)
