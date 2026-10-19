"""Pinned local loading and provenance shared by the richer experiment."""

import hashlib
import json
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten
from huggingface_hub import snapshot_download
from mlx_lm.tokenizer_utils import TokenizerWrapper
from mlx_lm.utils import load_model
from transformers import PreTrainedTokenizerFast

from .longcat import MODEL_ID,MODEL_REVISION


def load_native():
    mx.set_memory_limit(48_000_000_000);mx.set_cache_limit(128*1024**2)
    path=snapshot_download(MODEL_ID,revision=MODEL_REVISION,local_files_only=True,
        allow_patterns=['*.json','*.jinja','*.safetensors'])
    model,_=load_model(Path(path))
    tokenizer=TokenizerWrapper(PreTrainedTokenizerFast.from_pretrained(path,local_files_only=True),eos_token_ids=[2])
    return model,tokenizer


def frozen_digest(model):
    digest=hashlib.sha256()
    for name,array in tree_flatten(model.parameters()):
        if name.startswith('model.ngram_embeddings.residuals.') or name=='model.ngram_embeddings.scoped_vectors':continue
        digest.update(name.encode());digest.update(str(array.shape).encode());digest.update(str(array.dtype).encode())
        raw=array.view(mx.uint8).reshape(-1)
        for offset in range(0,raw.size,64*1024**2):digest.update(memoryview(np.array(raw[offset:offset+64*1024**2])))
        mx.clear_cache()
    return digest.hexdigest()


def snapshot_source(script,directory):
    out=Path(directory);out.mkdir()
    sources=[Path(script),*sorted(Path('engram_poc').glob('*.py'))];digest=hashlib.sha256()
    for path in sources:
        content=path.read_bytes();digest.update(path.name.encode());digest.update(content)
        (out/path.name).write_bytes(content)
    return digest.hexdigest()


def save(path,data):Path(path).write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n')


def append(path,data):
    with Path(path).open('a') as stream:stream.write(json.dumps(data,ensure_ascii=False)+'\n')
