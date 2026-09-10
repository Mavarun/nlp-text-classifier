"""Frozen pretrained sentence encoder wrapper (MiniLM via sentence-transformers).

Default model: ``sentence-transformers/all-MiniLM-L6-v2`` (384-d).
Encoder weights are **not** fine-tuned; callers fit a linear head on train
embeddings only. Supports an injectable ``encode_fn`` for offline / unit tests.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_EMBED_DIM = 384


EncodeFn = Callable[[Sequence[str]], np.ndarray]


def _hash_embed(texts: Sequence[str], dim: int = DEFAULT_EMBED_DIM) -> np.ndarray:
    """Deterministic offline fallback embeddings (no network / no weights).

    Not semantically meaningful — only for unit tests and documented offline
    dry-runs when the MiniLM weights are unavailable.
    """
    out = np.zeros((len(texts), dim), dtype=np.float32)
    for i, text in enumerate(texts):
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        # Expand digest into dim floats in [-1, 1]
        rng = np.random.RandomState(int.from_bytes(digest[:4], "little"))
        vec = rng.randn(dim).astype(np.float32)
        # Mix in a cheap bag-of-bytes signal so identical strings match
        for j, b in enumerate(digest):
            vec[j % dim] += (b / 255.0) - 0.5
        n = np.linalg.norm(vec) + 1e-8
        out[i] = vec / n
    return out


@dataclass
class PretrainedEncoder:
    """Thin wrapper: encode texts → float32 matrix (n, dim)."""

    model_name: str
    dim: int
    encode_fn: EncodeFn
    offline: bool = False

    def encode(self, texts: Sequence[str], *, batch_size: int = 32) -> np.ndarray:
        texts = list(texts)
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        # batch_size is advisory; injectable fns may ignore it
        emb = self.encode_fn(texts)
        emb = np.asarray(emb, dtype=np.float32)
        if emb.ndim != 2 or emb.shape[0] != len(texts):
            raise ValueError(f"encode_fn returned shape {emb.shape}, expected ({len(texts)}, dim)")
        return emb


def _load_sentence_transformer(model_name: str, cache_folder: str | None):
    from sentence_transformers import SentenceTransformer

    kwargs = {}
    if cache_folder:
        kwargs["cache_folder"] = cache_folder
    # local_files_only if explicitly requested via env
    if os.environ.get("NLP_TEXT_OFFLINE", "").lower() in {"1", "true", "yes"}:
        kwargs["local_files_only"] = True
    return SentenceTransformer(model_name, **kwargs)


def build_pretrained_encoder(
    model_name: str = DEFAULT_MODEL_NAME,
    *,
    cache_folder: str | None = None,
    encode_fn: EncodeFn | None = None,
    force_offline_hash: bool = False,
) -> PretrainedEncoder:
    """Build a frozen MiniLM encoder, or a hash fallback / injected encode_fn.

    Offline fallback: set ``force_offline_hash=True`` or env ``NLP_TEXT_FORCE_HASH=1``,
    or if sentence-transformers / model download fails and ``NLP_TEXT_ALLOW_HASH_FALLBACK=1``.
    """
    if encode_fn is not None:
        # Probe dim
        probe = np.asarray(encode_fn(["probe"]), dtype=np.float32)
        dim = int(probe.shape[-1]) if probe.ndim == 2 else DEFAULT_EMBED_DIM
        return PretrainedEncoder(
            model_name=model_name, dim=dim, encode_fn=encode_fn, offline=True
        )

    if force_offline_hash or os.environ.get("NLP_TEXT_FORCE_HASH", "").lower() in {
        "1",
        "true",
        "yes",
    }:
        return PretrainedEncoder(
            model_name=f"hash-fallback/{model_name}",
            dim=DEFAULT_EMBED_DIM,
            encode_fn=_hash_embed,
            offline=True,
        )

    try:
        model = _load_sentence_transformer(model_name, cache_folder)

        def _st_encode(texts: Sequence[str]) -> np.ndarray:
            arr = model.encode(
                list(texts),
                batch_size=32,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            return np.asarray(arr, dtype=np.float32)

        probe = _st_encode(["dimension probe"])
        return PretrainedEncoder(
            model_name=model_name,
            dim=int(probe.shape[-1]),
            encode_fn=_st_encode,
            offline=False,
        )
    except Exception as exc:  # noqa: BLE001 — documented fallback path
        if os.environ.get("NLP_TEXT_ALLOW_HASH_FALLBACK", "").lower() in {
            "1",
            "true",
            "yes",
        }:
            return PretrainedEncoder(
                model_name=f"hash-fallback/{model_name}",
                dim=DEFAULT_EMBED_DIM,
                encode_fn=_hash_embed,
                offline=True,
            )
        raise RuntimeError(
            f"Failed to load pretrained encoder {model_name!r}: {exc}. "
            "Set NLP_TEXT_FORCE_HASH=1 for offline hash embeddings, or "
            "NLP_TEXT_ALLOW_HASH_FALLBACK=1 to auto-fallback."
        ) from exc
