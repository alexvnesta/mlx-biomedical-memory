"""Canonical edge-phrase memory injected into a frozen MLX decoder.

This is an engineered Engram-inspired variant: deterministic phrase addresses,
learned vectors, projection and scalar gate. It is not a reproduction of DeepSeek
Engram. Addressing performs independent atomic lookups, never multi-hop traversal.
"""
from __future__ import annotations
import hashlib
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import create_attention_mask
from mlx.utils import tree_flatten


class PhraseIndex:
    def __init__(self, tokenizer, keys, slots=8192):
        self.slots = slots
        self.entries = []
        self.collisions_resolved = 0
        owners = {}
        for key in sorted(set(keys)):
            row = 1 + int.from_bytes(hashlib.blake2b(key.encode(), digest_size=8).digest(), "little") % (slots - 1)
            if len(owners) >= slots - 1:
                raise ValueError("memory table is full")
            # Open addressing avoids silently sharing an atomic fact's row.
            while row in owners:
                self.collisions_resolved += 1
                row = 1 + row % (slots - 1)
            owners[row] = key
            # Match the exact bracketed phrase, including the opening bracket.
            # Include context variants because BPE can merge a preceding space
            # into '['. Exclude ']' so a following '?' cannot change the match.
            for prefix in ("[", " [", "\n["):
                tokens = tokenizer.encode(prefix + key, add_special_tokens=False)
                self.entries.append((tokens, row, key))
        self.rows = {key: row for _, row, key in self.entries}

    def encode(self, token_ids):
        rows = [0] * len(token_ids)
        for phrase, row, _ in self.entries:
            size = len(phrase)
            for end in range(size, len(token_ids) + 1):
                if token_ids[end-size:end] == phrase:
                    rows[end-1] = row
        return rows


class ConditionalMemory(nn.Module):
    def __init__(self, slots, dimension, hidden_size):
        super().__init__()
        self.table = nn.Embedding(slots, dimension)
        self.projection = nn.Linear(dimension, hidden_size, bias=False)
        # Small, nonzero initialization permits table gradients on the first step.
        self.gate = mx.array(-2.0)

    def __call__(self, rows):
        return mx.sigmoid(self.gate) * self.projection(self.table(rows)) * (rows != 0)[..., None]


class MemoryModel(nn.Module):
    def __init__(self, backbone, slots=8192, dimension=64, layer=None):
        super().__init__()
        self.backbone = backbone
        self.backbone.freeze()
        self.layer = layer if layer is not None else len(backbone.model.layers) // 2
        self.memory = ConditionalMemory(slots, dimension, backbone.args.hidden_size)
        self.enabled = True

    def __call__(self, tokens, rows):
        # Full-prefix execution avoids incorrect n-gram history handling in a KV cache.
        core = self.backbone.model
        h = core.embed_tokens(tokens)
        mask = create_attention_mask(h, None)
        for i, layer in enumerate(core.layers):
            if i == self.layer and self.enabled:
                h = h + self.memory(rows).astype(h.dtype)
            h = layer(h, mask, None)
        # Keep the backbone's projection shape/order so disabling memory is
        # numerically identical to the stock model (GEMV/GEMM paths can differ).
        h = core.norm(h)
        if self.backbone.args.tie_word_embeddings:
            return core.embed_tokens.as_linear(h)[:, -1, :]
        return self.backbone.lm_head(h)[:, -1, :]


def fingerprint(module):
    """Full byte digest of all parameter tensors, not a sample."""
    digest = hashlib.sha256()
    for name, array in tree_flatten(module.parameters()):
        mx.eval(array)
        digest.update(name.encode())
        digest.update(str(array.shape).encode())
        digest.update(str(array.dtype).encode())
        # bfloat16 has no direct NumPy representation; view the actual bytes.
        digest.update(np.array(array.view(mx.uint8)).tobytes())
    return digest.hexdigest()
