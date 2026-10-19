import mlx.core as mx
import mlx.nn as nn

from engram_poc.decisive_model import FactCodec


def test_atomic_codec_is_row_local_and_receives_training_gradients():
    mx.random.seed(7);codec=FactCodec(hidden=32,width=8)
    fields=mx.array([[0,1,2,0,0,1,3],[1,2,4,1,1,0,0]])
    whole=codec(fields)
    assert whole.shape==(2,32)
    assert mx.allclose(codec(fields[::-1]),whole[::-1]).item()
    def loss(network):return mx.sum(network(fields)[:,0])
    value,grad=nn.value_and_grad(codec,loss)(codec)
    mx.eval(value,grad)
    assert mx.any(grad['projection']['weight']!=0).item()
    assert mx.any(grad['tables'][0]['weight']!=0).item()
