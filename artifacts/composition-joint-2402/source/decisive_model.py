"""Learned atomic tuple codec plus an adapted frozen LongCat reader.

The codec sees records only. It performs no traversal and receives no query or
answer. Text uses a larger LoRA rank so it has at least the candidate capacity.
Frozen lower-layer features may be cached for teacher-forced training only.
"""

from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import create_attention_mask
from mlx_lm.tuner.lora import LoRALinear
from mlx_lm.tokenizer_utils import TokenizerWrapper
from mlx_lm.utils import load_model
from transformers import PreTrainedTokenizerFast
from huggingface_hub import snapshot_download

from .disk_embeddings import offload_ngram_tables
from .longcat import MODEL_ID,MODEL_REVISION
from .longcat_editor import GradientSafeRouter


SPLIT_LAYER=10


class FactCodec(nn.Module):
    def __init__(self,hidden=3072,width=128):
        super().__init__()
        self.tables=[nn.Embedding(size,width) for size in [12,16,16,2,2,2,4]]
        self.projection=nn.Linear(7*width,hidden,bias=False)
        self.norm=nn.RMSNorm(hidden)
        self.gate=mx.array(-1.)

    def __call__(self,records):
        encoded=mx.concatenate([table(records[:,i]) for i,table in enumerate(self.tables)],axis=-1)
        return mx.sigmoid(self.gate)*self.norm(self.projection(encoded))


class Reader(nn.Module):
    def __init__(self,backbone,mode):
        super().__init__();self.backbone=backbone;self.mode=mode
        backbone.freeze()
        rank=16 if mode=='memory' else 32
        for layer in backbone.model.layers[SPLIT_LAYER:]:
            layer.mlp.router=GradientSafeRouter(layer.mlp.router)
            for attention in layer.self_attn:
                for name in ['q_a_proj','q_b_proj','kv_a_proj_with_mqa','o_proj']:
                    setattr(attention,name,LoRALinear.from_base(getattr(attention,name),r=rank,scale=8.,dropout=0.))
        if mode=='memory':self.memory=FactCodec(backbone.args.hidden_size)
        self._fields=None;self._positions=[];self._offset=0;self._enabled=True

    @property
    def layers(self):return self.backbone.layers

    def make_cache(self):return self.backbone.make_cache()

    def begin(self,fields,positions,enabled=True,offset=0):
        self._fields=fields;self._positions=positions;self._enabled=enabled;self._offset=offset

    def prefix(self,tokens):
        core=self.backbone.model;h=core.ngram_embeddings(tokens)
        mask=create_attention_mask(h,None,return_array=True)
        for layer in core.layers[:SPLIT_LAYER]:h=layer(h,mask,cache=None)
        return mx.stop_gradient(h)

    def tail(self,h,fields,positions,*,loss_start=None,cache=None,offset=0):
        core=self.backbone.model
        if self.mode=='memory' and self._enabled and positions:
            # Injection is determined only by exact record slots and values.
            if any(offset<=p<offset+h.shape[1] for p in positions):
                vectors=self.memory(fields).astype(h.dtype)
                mask=(mx.arange(h.shape[1])[:,None]+offset==mx.array(positions)[None,:]).astype(h.dtype)
                h=h+(mask@vectors)[None,:,:]
        mask=create_attention_mask(h,cache[SPLIT_LAYER+1][0] if cache else None,return_array=True)
        for index,layer in enumerate(core.layers[SPLIT_LAYER:],SPLIT_LAYER):
            h=layer(h,mask,cache=cache[index+1] if cache else None)
        h=core.norm(h)
        if loss_start is not None:h=h[:,loss_start:]
        return self.backbone.lm_head(h)

    def __call__(self,tokens,cache=None):
        core=self.backbone.model;h=core.ngram_embeddings(tokens,cache=cache[0] if cache else None)
        mask=create_attention_mask(h,cache[1][0] if cache else None,return_array=True)
        for index,layer in enumerate(core.layers[:SPLIT_LAYER]):h=layer(h,mask,cache=cache[index+1] if cache else None)
        h=mx.stop_gradient(h)
        result=self.tail(h,self._fields,self._positions,cache=cache,offset=self._offset if cache else 0)
        if cache:self._offset+=tokens.shape[1]
        return result


def load_reader(mode,*,tail_only=False,verify_rows=False):
    path=Path(snapshot_download(MODEL_ID,revision=MODEL_REVISION,local_files_only=True,
        allow_patterns=['*.json','*.jinja','*.safetensors']))
    backbone,_=load_model(path,lazy=True)
    if tail_only:
        backbone.model.ngram_embeddings=nn.Identity()
        for i in range(SPLIT_LAYER):backbone.model.layers[i]=nn.Identity()
    else:offload_ngram_tables(backbone,path,verify_rows=verify_rows)
    model=Reader(backbone,mode)
    tokenizer=TokenizerWrapper(PreTrainedTokenizerFast.from_pretrained(path,local_files_only=True),eos_token_ids=[2])
    return model,tokenizer
