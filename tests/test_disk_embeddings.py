import mlx.core as mx
import mlx.nn as nn
import pytest

from engram_poc.disk_embeddings import DiskQuantizedEmbedding


def test_disk_quantized_rows_preserve_native_values_and_input_shape(tmp_path):
    native=nn.QuantizedEmbedding(41,128,group_size=64,bits=4)
    mx.eval(native.parameters());path=tmp_path/'rows.safetensors'
    mx.save_safetensors(str(path),{'e.weight':native.weight,'e.scales':native.scales,'e.biases':native.biases})
    disk=DiskQuantizedEmbedding(path,'e')
    for ids in [mx.array([[0,40,3,3]]),mx.array([[1,2],[39,0]])]:
        assert mx.array_equal(native(ids),disk(ids)).item()
    assert not disk.parameters()
    with pytest.raises(IndexError):disk(mx.array([-1]))
