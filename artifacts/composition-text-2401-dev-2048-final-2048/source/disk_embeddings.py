"""Read-only native quantized embedding rows without resident full tables.

Only storage changes. Hashing, quantization parameters and dequantization are
the native MLX operations. No graph traversal or answer computation occurs here.
"""

import json
from pathlib import Path
import struct

import mlx.core as mx
import mlx.nn as nn
import numpy as np


DTYPES={'U32':np.uint32,'U16':np.uint16,'BF16':np.uint16,'F32':np.float32,'F16':np.float16}


class TensorRows:
    def __init__(self,path,name):
        with Path(path).open('rb') as stream:
            size=struct.unpack('<Q',stream.read(8))[0]
            metadata=json.loads(stream.read(size))[name]
        self.dtype=metadata['dtype'];self.shape=tuple(metadata['shape'])
        self.array=np.memmap(path,mode='r',dtype=DTYPES[self.dtype],shape=self.shape,
            offset=8+size+metadata['data_offsets'][0])

    def take(self,indices):
        value=mx.array(np.array(self.array[indices],copy=True))
        return value.view(mx.bfloat16) if self.dtype=='BF16' else value


class DiskQuantizedEmbedding(nn.Module):
    def __init__(self,path,prefix,group_size=64,bits=4,mode='affine'):
        super().__init__()
        self._weight=TensorRows(path,prefix+'.weight')
        self._scales=TensorRows(path,prefix+'.scales')
        self._biases=TensorRows(path,prefix+'.biases')
        self.group_size=group_size;self.bits=bits;self.mode=mode
        self.num_embeddings=self._weight.shape[0]
        self.dims=self._weight.shape[-1]*32//bits
        self.freeze()

    def __call__(self,indices):
        ids=np.array(indices).astype(np.int64,copy=False)
        if np.any(ids<0) or np.any(ids>=self.num_embeddings):raise IndexError('Embedding row out of range')
        return mx.dequantize(self._weight.take(ids),self._scales.take(ids),self._biases.take(ids),
            group_size=self.group_size,bits=self.bits,mode=self.mode)


def offload_ngram_tables(model,path,*,verify_rows=True):
    """Call immediately after lazy model loading, before evaluating parameters."""
    path=Path(path);index=json.loads((path/'model.safetensors.index.json').read_text())['weight_map']
    receipts=[]
    for i,native in enumerate(model.model.ngram_embeddings.embedders):
        prefix=f'model.ngram_embeddings.embedders.{i}'
        shard=index[prefix+'.weight']
        assert all(index[prefix+'.'+name]==shard for name in ['scales','biases'])
        disk=DiskQuantizedEmbedding(path/shard,prefix,native.group_size,native.bits,native.mode)
        if verify_rows:
            ids=mx.array([[0,1,17,disk.num_embeddings//2,disk.num_embeddings-1]])
            reference=native(ids);actual=disk(ids);mx.eval(reference,actual)
            assert mx.array_equal(reference,actual).item(),prefix+' native row parity'
        receipts.append(dict(table=prefix,rows=disk.num_embeddings,dims=disk.dims,
            shard=shard,sampled_rows_exact=verify_rows))
        model.model.ngram_embeddings.embedders[i]=disk
        del native;mx.clear_cache()
    model.freeze()
    return receipts
