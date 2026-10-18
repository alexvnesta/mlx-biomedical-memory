from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from engram_poc.longcat_editor import EditableMemory, GradientSafeRouter, allocate_updates


class Router(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Linear(8, 4)
        self.e_score_correction_bias = mx.array([0., 0.1, -0.1, 0.])
        self.top_k = 2
        self.norm_topk_prob = False
        self.routed_scaling_factor = 6.

    def __call__(self, x):
        scores = mx.softmax(self.classifier(x), axis=-1)
        indices = mx.argpartition(scores+self.e_score_correction_bias, kth=-2, axis=-1)[..., -2:]
        return indices, mx.take_along_axis(scores, indices, axis=-1)*6


def test_router_preserves_forward_and_allows_input_gradients():
    native = Router()
    safe = GradientSafeRouter(native)
    x = mx.array([[1., 2, 3, 4, 5, 6, 7, 8]])
    a, b = native(x), safe(x)
    assert all(mx.array_equal(u, v).item() for u, v in zip(a, b))
    gradient = mx.grad(lambda z: mx.sum(safe(z)[1]**2))(x)
    assert mx.all(mx.isfinite(gradient)).item()
    assert mx.linalg.norm(gradient).item() > 0


def test_joint_allocation_accounts_for_shared_keys_and_regularization():
    # Two expressions share their bigram, with different trigram keys.
    target = dict(current=np.zeros((2, 2), dtype=np.float32),
        targets=np.array([[2., 1.], [1., 3.]], dtype=np.float32),
        key_sets=[{2: (1, 3), 3: (4, 1, 3)}, {2: (1, 3), 3: (5, 1, 3)}])
    updates, report = allocate_updates([target], lambda_reuse=0, lambda_ridge=0.1)
    a = np.array([[1., 1., 0.], [1., 0., 1.]], dtype=np.float32)
    expected = np.linalg.solve(a.T@a+0.1*np.eye(3), a.T@target['targets'])
    actual = np.stack([np.array(v) for v in updates.values()])
    assert np.allclose(actual, expected, atol=1e-5)
    assert report['keys'] == 3


class Native(nn.Module):
    vocab_size, hidden_size = 32, 4
    def __call__(self, ids, cache=None):
        return mx.zeros((*ids.shape, 4))


def test_new_updates_rollback_and_probe_only_changes_selected_positions():
    memory = EditableMemory(Native())
    before = memory.snapshot()
    memory.add_updates({(2, (3, 4)): mx.array([1., 2, 3, 4])})
    ids = mx.array([[3, 4, 5]])
    assert memory(ids)[0, 1].tolist() == [1., 2, 3, 4]
    memory._probe_delta = mx.ones(4)
    memory._probe_position = 2
    assert memory(ids)[0, 2].tolist() == [1., 1, 1, 1]
    memory._probe_delta = None
    memory.restore(before)
    assert mx.all(memory(ids) == 0).item()
