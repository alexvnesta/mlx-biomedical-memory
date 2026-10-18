import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx_lm.models.cache import ArraysCache

from engram_poc.scoped_memory import ScopedMemory


class Native(nn.Module):
    vocab_size,hidden_size=32,4
    def __call__(self,ids,cache=None):
        if cache is not None:
            previous=cache[0]
            full=ids if previous is None else mx.concatenate([previous,ids],axis=-1)
            cache[0]=full[:,-3:]
        return mx.broadcast_to(ids[...,None].astype(mx.float32),(*ids.shape,4))


def test_same_entity_different_scope_stays_separate_and_cached_chunks_match():
    memory=ScopedMemory(Native())
    memory.store(['C/human','C/mouse'],[mx.array([1.,2,3,4]),mx.array([4.,3,2,1])],[2,1])
    ids=mx.array([[3,4,5,6]])
    memory.begin('scoped',[('C/human',2)])
    full=memory(ids)
    assert full[0,2].tolist()==[6.,7,8,9]
    memory.begin('scoped',[('C/mouse',2)])
    assert memory(ids)[0,2].tolist()==[9.,8,7,6]
    memory.begin('scoped',[('C/human',2)])
    cache=ArraysCache(size=1)
    chunked=mx.concatenate([memory(ids[:,:2],cache),memory(ids[:,2:],cache)],axis=1)
    assert mx.array_equal(full,chunked).item()
    memory.begin('native')
    assert mx.array_equal(memory(ids),Native()(ids)).item()
    memory.begin('shuffled',[('C/human',2)])
    assert memory(ids)[0,2].tolist()==[9.,8,7,6]


def test_bank_rejects_misaligned_or_nonfinite_rows():
    memory=ScopedMemory(Native())
    with pytest.raises(ValueError,match='One vector'):
        memory.store(['C'],[],[1])
    with pytest.raises(ValueError,match='shape'):
        memory.store(['C'],[mx.zeros((3,))],[1])
    with pytest.raises(ValueError,match='Nonfinite'):
        memory.store(['C'],[mx.array([float('nan'),0,0,0])],[1])
