import hashlib
import json

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx_lm.models.cache import ArraysCache

from engram_poc.longcat import EditedNgramEmbedding, MODEL_REVISION


class TinyNativeEmbedding(nn.Module):
    vocab_size = 32
    hidden_size = 4

    def __call__(self, ids, cache=None):
        if cache is not None:
            previous = cache[0]
            full = ids if previous is None else mx.concatenate([previous, ids], axis=-1)
            cache[0] = full[:, -3:]
        return mx.broadcast_to(ids[..., None].astype(mx.float32), (*ids.shape, 4))


def test_edits_survive_cached_generation_and_disabled_wrapper_is_exact(tmp_path):
    arrays = {
        "2": mx.array([[0., 0, 0, 0], [1., 2, 3, 4]]),
        "3": mx.array([[0., 0, 0, 0], [4., 3, 2, 1]]),
        "4": mx.array([[0., 0, 0, 0], [2., 2, 2, 2]]),
    }
    path = tmp_path / "updates.safetensors"
    mx.save_safetensors(str(path), arrays)
    (tmp_path / "manifest.json").write_text(json.dumps(dict(
        format="engramedit-mlx-v1", model_revision=MODEL_REVISION,
        updates_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        keys={"2": [[4, 5]], "3": [[3, 4, 5]], "4": [[3, 4, 5, 6]]})))
    wrapper = EditedNgramEmbedding(TinyNativeEmbedding(), tmp_path)
    tokens = mx.array([[3, 4, 5, 6]])
    full = wrapper(tokens)
    cache = ArraysCache(size=1)
    chunked = mx.concatenate([wrapper(tokens[:, :2], cache), wrapper(tokens[:, 2:], cache)], axis=1)
    assert mx.array_equal(full, chunked).item()
    assert full[0, 2].tolist() == [10., 10., 10., 10.]
    assert full[0, 3].tolist() == [8., 8., 8., 8.]
    wrapper.enabled = False
    assert mx.array_equal(wrapper(tokens), wrapper.base_embedding(tokens)).item()
    path.write_bytes(path.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        EditedNgramEmbedding(TinyNativeEmbedding(), tmp_path)
