"""Replay native LongCat memory edits on MLX, including cached generation.

This ports the released exact-key additive update interface, not the target
optimization or regularized solver. The pretrained native memory stays intact.
Adapted from EngramEdit/memory.py, ModalityDance/EngramEdit commit
fb4089be4eec2575847d7e71342ff545ccb70508. MIT attribution in third_party/engramedit.
"""

import json
import hashlib
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .longcat_keys import LENGTHS, activated_rows


MODEL_ID = "mlx-community/LongCat-Flash-Lite-4bit"
MODEL_REVISION = "0435b3bca4418830335d784317fe7f67f8383253"
STATE_ID = "ModalityDance/EngramEdit-LongCat-Flash-Lite-MQuAKE-3K"
STATE_REVISION = "6222d320052e349af9cf76417ebd329da8b5350b"
UPSTREAM_COMMIT = "fb4089be4eec2575847d7e71342ff545ccb70508"
MQUAKE_COMMIT = "fb43dadc2d8cd19d08ce81c63d957b59deb3f3cd"


class EditedNgramEmbedding(nn.Module):
    """Add projected exact-key residuals after the native embedding average."""

    def __init__(self, base_embedding, state_dir):
        super().__init__()
        self.base_embedding = base_embedding
        manifest = json.loads((Path(state_dir) / "manifest.json").read_text())
        if manifest["format"] != "engramedit-mlx-v1":
            raise ValueError("Unsupported memory format")
        if manifest["model_revision"] != MODEL_REVISION:
            raise ValueError("Memory was prepared for a different model revision")
        updates_path = Path(state_dir) / "updates.safetensors"
        if hashlib.sha256(updates_path.read_bytes()).hexdigest() != manifest["updates_sha256"]:
            raise ValueError("Memory checksum does not match preparation manifest")
        self._keys = {length: {} for length in LENGTHS}
        for length in LENGTHS:
            for row, key in enumerate(manifest["keys"][str(length)], 1):
                if len(key) != length or any(type(x) is not int or x < 0 or x >= base_embedding.vocab_size for x in key):
                    raise ValueError("Invalid memory key")
                if tuple(key) in self._keys[length]:
                    raise ValueError("Duplicate memory key")
                self._keys[length][tuple(key)] = row
        weights = mx.load(str(updates_path))
        self.residuals = {}
        for length in LENGTHS:
            rows = weights[str(length)]
            expected = (len(self._keys[length]) + 1, base_embedding.hidden_size)
            if rows.shape != expected or not mx.all(mx.isfinite(rows)).item():
                raise ValueError("Invalid memory residuals")
            if mx.any(rows[0] != 0).item():
                raise ValueError("Null memory row must be zero")
            self.residuals[str(length)] = rows
        self.enabled = True
        self._hit_counts = {length: 0 for length in LENGTHS}
        self.freeze()

    def __call__(self, input_ids, cache=None):
        # Snapshot the old suffix before the native embedding advances its cache.
        previous = cache[0] if cache is not None else None
        output = self.base_embedding(input_ids, cache=cache)
        if not self.enabled:
            return output
        token_rows = np.array(input_ids).tolist()
        history = np.array(previous).tolist() if previous is not None else [[] for _ in token_rows]
        indices = activated_rows(token_rows, history, self._keys)
        residual = mx.zeros_like(output)
        for length in LENGTHS:
            self._hit_counts[length] += sum(row > 0 for batch in indices[length] for row in batch)
            # Preserve the upstream per-length accumulation in the output dtype.
            residual = residual + self.residuals[str(length)][mx.array(indices[length])].astype(output.dtype)
        return output + residual


def install_memory(model, state_dir):
    if model.model_type != "longcat_flash_ngram":
        raise ValueError("Expected native LongCat-Flash-Lite n-gram architecture")
    model.freeze()
    wrapper = EditedNgramEmbedding(model.model.ngram_embeddings, state_dir)
    model.model.ngram_embeddings = wrapper
    return wrapper
