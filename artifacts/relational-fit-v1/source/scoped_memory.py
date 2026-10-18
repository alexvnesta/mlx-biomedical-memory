"""Explicit tuple-addressed latent residuals, an engineered interface.

This is not native n-gram addressing or an exact EngramEdit implementation.
The same independently fitted deltas can be allocated to subject n-grams or
stored independently by (entity, relation, context). Routing supplies no answer.
"""

import hashlib
import json
from pathlib import Path

import mlx.core as mx

from .longcat import MODEL_REVISION
from .longcat_editor import EditableMemory,install_editable


class ScopedMemory(EditableMemory):
    def __init__(self,native):
        super().__init__(native)
        self.scoped_vectors=mx.zeros((1,native.hidden_size),dtype=mx.float32)
        self._scope_index={};self._placements=[];self._offset=0;self._mode='native'
        self._shuffle=[]
        self.freeze()

    def store(self,keys,vectors,permutation):
        if len(keys)!=len(set(keys)):raise ValueError('Duplicate scoped address')
        if sorted(permutation)!=list(range(1,len(keys)+1)):raise ValueError('Invalid row permutation')
        self._scope_index={key:i for i,key in enumerate(keys,1)}
        self.scoped_vectors=mx.stack([mx.zeros((self.base_embedding.hidden_size,)),
            *[v.astype(mx.bfloat16).astype(mx.float32) for v in vectors]])
        self._shuffle=[0,*permutation];mx.eval(self.scoped_vectors);self.freeze()

    def begin(self,mode,placements=()):
        if mode not in ['native','ngram','scoped','shuffled']:raise ValueError(mode)
        self._mode=mode;self.enabled=mode=='ngram';self._offset=0
        self._placements=[]
        for key,position in placements:
            if key not in self._scope_index:raise KeyError(key)
            if position<0:raise ValueError('Negative injection position')
            row=self._scope_index[key]
            self._placements.append((self._shuffle[row] if mode=='shuffled' else row,position))

    def __call__(self,ids,cache=None):
        output=super().__call__(ids,cache)
        offset=self._offset if cache is not None else 0
        if self._mode in ['scoped','shuffled']:
            if ids.shape[0]!=1:raise ValueError('Scoped generation supports one sequence per request')
            positions=mx.arange(ids.shape[-1])+offset
            for row,position in self._placements:
                output=output+(positions==position)[None,:,None]*self.scoped_vectors[row].astype(output.dtype)
        if cache is not None:self._offset+=ids.shape[-1]
        return output

    def save_scoped(self,directory,metadata):
        out=Path(directory);out.mkdir(parents=True,exist_ok=True)
        self.save(out/'ngram',metadata)
        path=out/'scoped.safetensors';mx.save_safetensors(str(path),{'vectors':self.scoped_vectors})
        manifest=dict(format='scoped-residual-mlx-v1',model_revision=MODEL_REVISION,
            vectors_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),keys=list(self._scope_index),
            permutation=self._shuffle,metadata=metadata)
        (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


def install_scoped(model):
    editable=install_editable(model)
    memory=ScopedMemory(editable.base_embedding)
    model.model.ngram_embeddings=memory;model.freeze()
    return memory
