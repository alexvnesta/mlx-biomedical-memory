"""Compare the MLX residual wrapper with the actual pinned upstream PyTorch code.

No large model is loaded. Both wrappers receive identical synthetic native
embedding outputs, actual released edits, EOS boundaries, and chunked inputs.
This isolates replay fidelity from backbone quantization and model kernels.
"""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import types

import mlx.core as mx
from mlx_lm.models.cache import ArraysCache
import mlx.nn as nn
import numpy as np
import torch
from huggingface_hub import hf_hub_download

from engram_poc.longcat import EditedNgramEmbedding, STATE_ID, STATE_REVISION, UPSTREAM_COMMIT


class MlxBase(nn.Module):
    vocab_size = 131072
    hidden_size = 3072

    def __call__(self, ids, cache=None):
        if cache is not None:
            old = cache[0]
            context = ids if old is None else mx.concatenate([old, ids], axis=-1)
            cache[0] = context[:, -3:]
        value = (ids % 17).astype(mx.float32) / 16
        return mx.broadcast_to(value[..., None], (*ids.shape, 3072)).astype(mx.bfloat16)


class TorchBase(torch.nn.Module):
    config = types.SimpleNamespace(eos_token_id=2, vocab_size=131072, hidden_size=3072,
                                   emb_neighbor_num=4)

    def forward(self, ids, ngram_context=None):
        value = (ids % 17).float() / 16
        return value[..., None].expand(*ids.shape, 3072).to(torch.bfloat16)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--state", type=Path, default=Path("data/processed/longcat"))
    parser.add_argument("--output", type=Path, default=Path("runs/longcat-reference-parity.json"))
    args = parser.parse_args()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.upstream_root, text=True).strip()
    if commit != UPSTREAM_COMMIT:
        raise ValueError("Upstream checkout is not at the audited revision")
    # Isolate this file's only relative import. No other upstream modules run.
    package = types.ModuleType("reference_engramedit")
    package.__path__ = []
    data = types.ModuleType("reference_engramedit.data")
    data.NGRAM_LENGTHS = (2, 3, 4)
    sys.modules[package.__name__] = package
    sys.modules[data.__name__] = data
    source = args.upstream_root / "EngramEdit/memory.py"
    spec = importlib.util.spec_from_file_location("reference_engramedit.memory", source)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    path = hf_hub_download(STATE_ID, "engramedit_state.pt", revision=STATE_REVISION, local_files_only=True)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    torch_wrapper = reference.EngramEditEmbedding(TorchBase())
    torch_wrapper.load_snapshot(payload)
    mlx_wrapper = EditedNgramEmbedding(MlxBase(), args.state)
    manifest = json.loads((args.state / "manifest.json").read_text())
    tokens = [1]
    for length in (2, 3, 4):
        for key in manifest["keys"][str(length)][:4]:
            tokens.extend(key)
            tokens.append(2)
    ids = mx.array([tokens])
    torch_ids = torch.tensor([tokens], dtype=torch.long)
    expected = torch_wrapper(torch_ids).float().numpy()
    full = np.array(mlx_wrapper(ids).astype(mx.float32))
    cache = ArraysCache(size=1)
    pieces = []
    torch_pieces = []
    history = None
    for start in range(0, len(tokens), 2):
        chunk = ids[:, start:start+2]
        pieces.append(mlx_wrapper(chunk, cache))
        torch_chunk = torch_ids[:, start:start+2]
        torch_pieces.append(torch_wrapper(torch_chunk, ngram_context=history))
        history = torch_chunk if history is None else torch.cat([history, torch_chunk], dim=-1)
        history = history[:, -3:]
    chunked = np.array(mx.concatenate(pieces, axis=1).astype(mx.float32))
    torch_chunked = torch.cat(torch_pieces, dim=1).float().numpy()
    result = dict(upstream_commit=commit, upstream_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        dtype="bfloat16", sequence_tokens=len(tokens), hidden_size=3072,
        full_exact=bool(np.array_equal(full, expected)),
        cached_exact=bool(np.array_equal(chunked, torch_chunked)),
        mlx_chunked_equals_full=bool(np.array_equal(chunked, full)),
        max_absolute_error=float(np.max(np.abs(full-expected))),
        scope="Residual replay only; synthetic identical base outputs with actual published vectors. Does not establish BF16/4-bit backbone equivalence.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))
    if not all(result[k] for k in ("full_exact", "cached_exact", "mlx_chunked_equals_full")):
        raise RuntimeError("Upstream residual replay parity failed")


if __name__ == "__main__":
    main()
