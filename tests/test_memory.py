import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.qwen3 import Model, ModelArgs
from mlx.utils import tree_flatten
from engram_poc.memory import MemoryModel, fingerprint


def tiny_model():
    args = ModelArgs(
        model_type="qwen3",
        hidden_size=16,
        num_hidden_layers=2,
        intermediate_size=32,
        num_attention_heads=2,
        rms_norm_eps=1e-6,
        vocab_size=32,
        num_key_value_heads=1,
        max_position_embeddings=256,
        rope_theta=10000,
        head_dim=8,
        tie_word_embeddings=True,
    )
    return Model(args)


def test_memory_disabled_matches_backbone_and_backbone_has_no_gradients():
    base = tiny_model()
    m = MemoryModel(base, slots=8, dimension=2, layer=1)
    tokens = mx.array([[1, 2, 3, 4]])
    rows = mx.array([[0, 1, 0, 2]])
    m.enabled = False
    assert mx.allclose(m(tokens, rows), base(tokens)[:, -1, :], atol=1e-5).item()
    assert all(
        not n.startswith("backbone.") for n, _ in tree_flatten(m.trainable_parameters())
    )


def test_gradient_reaches_memory_through_frozen_backbone():
    m = MemoryModel(tiny_model(), slots=8, dimension=2, layer=1)
    before = fingerprint(m.backbone)
    tokens, rows = mx.array([[1, 2, 3, 4]]), mx.array([[0, 1, 0, 2]])
    loss, grad = nn.value_and_grad(m, lambda model: mx.sum(model(tokens, rows) ** 2))(m)
    flat = dict(tree_flatten(grad))
    assert any(
        mx.any(mx.abs(g) > 0).item()
        for name, g in flat.items()
        if name.startswith("memory.")
    )
    assert fingerprint(m.backbone) == before


def test_null_addresses_do_not_inject_trainable_background():
    base = tiny_model()
    m = MemoryModel(base, slots=8, dimension=2, layer=1)
    tokens = mx.array([[1, 2, 3]])
    assert mx.allclose(
        m(tokens, mx.zeros(tokens.shape, dtype=mx.int32)),
        base(tokens)[:, -1, :],
        atol=1e-5,
    ).item()
