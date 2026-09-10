"""Encoder wrapper: shapes, determinism, offline hash path (no network)."""

import numpy as np

from nlp_text.pretrained_encoder import (
    DEFAULT_EMBED_DIM,
    build_pretrained_encoder,
)


def test_hash_encoder_shapes_and_determinism():
    enc = build_pretrained_encoder(force_offline_hash=True)
    texts = ["hello world", "other document", "hello world"]
    a = enc.encode(texts)
    b = enc.encode(texts)
    assert a.shape == (3, DEFAULT_EMBED_DIM)
    assert a.dtype == np.float32
    np.testing.assert_allclose(a, b)
    # identical strings → identical vectors
    np.testing.assert_allclose(a[0], a[2])
    # different strings → different vectors (hash collision extremely unlikely)
    assert not np.allclose(a[0], a[1])


def test_injected_encode_fn():
    def fake(texts):
        return np.arange(len(texts) * 8, dtype=np.float32).reshape(len(texts), 8)

    enc = build_pretrained_encoder(encode_fn=fake)
    out = enc.encode(["a", "b"])
    assert out.shape == (2, 8)
