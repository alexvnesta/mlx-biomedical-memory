"""Quantization-aware MLX adaptation of EngramEdit target fitting and allocation.

Algorithm adapted from ModalityDance/EngramEdit, fb4089be4eec2575847d7e71342ff545ccb70508,
MIT; see third_party/engramedit. Expressions and calibration corpus are explicit
inputs. Local calibration is not the paper's Wikipedia frequency resource.
"""

from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .longcat import EditedNgramEmbedding, MODEL_REVISION
from .longcat_keys import LENGTHS


class GradientSafeRouter(nn.Module):
    """Preserve native forward values; stop derivatives of discrete expert IDs.

    MLX-LM 0.31.3's LongCat inference router does not stop these integer
    derivatives. PyTorch's discrete top-k selection already has this behavior.
    Selected softmax weights still receive gradients with respect to inputs.
    """
    def __init__(self, native):
        super().__init__()
        self.native = native

    def __call__(self, hidden):
        router = self.native
        scores = mx.softmax(router.classifier(hidden), axis=-1)
        corrected = scores + router.e_score_correction_bias
        indices = mx.stop_gradient(mx.argpartition(corrected, kth=-router.top_k, axis=-1)[..., -router.top_k:])
        weights = mx.take_along_axis(scores, indices, axis=-1)
        if router.norm_topk_prob:
            weights = weights/(mx.sum(weights, axis=-1, keepdims=True)+1e-20)
        return indices, (weights*router.routed_scaling_factor).astype(hidden.dtype)


class EditableMemory(EditedNgramEmbedding):
    def __init__(self, native):
        nn.Module.__init__(self)
        self.base_embedding = native
        self._keys = {length: {} for length in LENGTHS}
        self._stored = {length: {} for length in LENGTHS}
        self.residuals = {str(length): mx.zeros((1, native.hidden_size)) for length in LENGTHS}
        self.enabled = True
        self._hit_counts = {length: 0 for length in LENGTHS}
        self._probe_delta = None
        self._probe_position = 0
        self.freeze()

    def __call__(self, ids, cache=None):
        output = super().__call__(ids, cache)
        if self._probe_delta is not None:
            if cache is not None:
                raise ValueError("Target fitting must not reuse generation caches")
            positions = self._probe_position
            if isinstance(positions, int):
                positions = mx.full((ids.shape[0],), positions, dtype=mx.int32)
            mask = (mx.arange(ids.shape[-1])[None, :] == positions[:, None])[..., None]
            output = output + mask * self._probe_delta.astype(output.dtype)
        return output

    def add_updates(self, updates):
        for (length, key), vector in updates.items():
            current = self._stored[length].get(tuple(key), mx.zeros_like(vector))
            self._stored[length][tuple(key)] = (current.astype(mx.bfloat16) + vector.astype(mx.bfloat16)).astype(mx.float32)
        self.rebuild()

    def rebuild(self):
        for length in LENGTHS:
            keys = sorted(self._stored[length])
            self._keys[length] = {key: row for row, key in enumerate(keys, 1)}
            self.residuals[str(length)] = mx.stack([
                mx.zeros((self.base_embedding.hidden_size,)),
                *[self._stored[length][key] for key in keys],
            ])
        mx.eval(self.residuals)
        self.freeze()

    def snapshot(self):
        return {length: dict(table) for length, table in self._stored.items()}

    def restore(self, snapshot):
        self._stored = {length: dict(table) for length, table in snapshot.items()}
        self.rebuild()

    def save(self, directory, metadata):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "updates.safetensors"
        mx.save_safetensors(str(path), self.residuals)
        manifest = dict(format="engramedit-mlx-v1", model_revision=MODEL_REVISION,
            updates_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            keys={str(length): [list(key) for key in sorted(self._keys[length])] for length in LENGTHS},
            editor=metadata)
        (directory / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")


def install_editable(model):
    if model.model_type != "longcat_flash_ngram":
        raise ValueError("Expected native LongCat n-gram architecture")
    model.freeze()
    for layer in model.model.layers:
        if not isinstance(layer.mlp.router, GradientSafeRouter):
            layer.mlp.router = GradientSafeRouter(layer.mlp.router)
    model.freeze()
    memory = EditableMemory(model.model.ngram_embeddings)
    model.model.ngram_embeddings = memory
    return memory


def position_and_keys(tokenizer, template, subject):
    if template.count("{}") != 1:
        raise ValueError("Each template needs exactly one subject placeholder")
    prefix = template.split("{}")[0]
    position = len(tokenizer.encode(prefix + subject, add_special_tokens=False)) - 1
    tokens = tokenizer.encode(template.format(subject), add_special_tokens=False)
    if not 0 <= position < len(tokens):
        raise ValueError("Invalid subject position")
    keys = {}
    for length in LENGTHS:
        start = position-length+1
        if start >= 0:
            key = tuple(tokens[start:position+1])
            if tokenizer.eos_token_id not in key[:-1]:
                keys[length] = key
    return position, keys


def log_probs(logits):
    values = logits.astype(mx.float32)
    return values - mx.logsumexp(values, axis=-1, keepdims=True)


def fit_target(model, memory, tokenizer, request, *, steps=25, lr=0.5,
               contexts=("{}",), expressions=None, log=None):
    """Fit one shared 3072-vector, accumulating serial-example gradients.

    Serial accumulation avoids padding/mask differences and the large batch used
    on CUDA. The objective averages rewriting losses, adds current||initial KL,
    and applies the paper's norm penalty and clamp. No base parameters are optimized.
    """
    started = time.perf_counter()
    subject, template = request["subject"], request["prompt"]
    sources = list(dict.fromkeys([template, *(expressions or [])]))
    target_text = " " + request["target"].lstrip()
    target = tokenizer.encode(target_text, add_special_tokens=False)
    if not target:
        raise ValueError("Empty editing target")
    rows = []
    key_sets = []
    initials = []
    for source in sources:
        _, keys = position_and_keys(tokenizer, source, subject)
        for context in contexts:
            expression = context.format(source)
            position, _ = position_and_keys(tokenizer, expression, subject)
            prompt_ids = tokenizer.encode(expression.format(subject), add_special_tokens=False)
            ids = mx.array([prompt_ids + target[:-1]])
            initial = memory(ids)[0, position].astype(mx.float32)
            mx.eval(initial)
            initials.append(initial)
            rows.append((ids, position, len(prompt_ids)-1))
            key_sets.append(keys)
    kl_template = "{} is a"
    kl_pos, _ = position_and_keys(tokenizer, kl_template, subject)
    kl_ids = mx.array([tokenizer.encode(kl_template.format(subject), add_special_tokens=False)])
    initial_kl = log_probs(model(kl_ids)[0, kl_pos])
    mx.eval(initial_kl)
    norm = float(mx.linalg.norm(initials[0]).item())
    denominator = max(norm*norm, 1e-12)
    delta = mx.zeros((model.args.hidden_size,), dtype=mx.float32)
    moment, variance = mx.zeros_like(delta), mx.zeros_like(delta)
    target_ids = mx.array(target)
    history = []

    groups = {}
    for ids, position, offset in rows:
        groups.setdefault((ids.shape[-1], offset), []).append((ids, position))
    batches = []
    for (_, offset), group in groups.items():
        batches.append((mx.concatenate([x[0] for x in group], axis=0),
            mx.array([x[1] for x in group]), offset, len(group)/len(rows)))

    def rewriting_loss(value, ids, position, offset):
        memory._probe_delta = value
        memory._probe_position = position
        lp = log_probs(model(ids)[:, offset:offset+len(target)])
        indices = mx.broadcast_to(target_ids[None, :, None], (ids.shape[0], len(target), 1))
        return -mx.mean(mx.take_along_axis(lp, indices, axis=-1))

    def preservation_loss(value):
        memory._probe_delta = value
        memory._probe_position = kl_pos
        current = log_probs(model(kl_ids)[0, kl_pos])
        return 0.0625 * mx.sum(mx.exp(current)*(current-initial_kl))

    try:
        for step in range(steps):
            nll, grad = mx.array(0.0), mx.zeros_like(delta)
            for ids, position, offset, weight in batches:
                value, derivative = mx.value_and_grad(rewriting_loss)(delta, ids, position, offset)
                mx.eval(value, derivative)
                nll = nll + value*weight
                grad = grad + derivative*weight
            kl, kl_grad = mx.value_and_grad(preservation_loss)(delta)
            safe_norm = mx.sqrt(mx.sum(delta*delta) + 1e-12)
            penalty = 0.001*safe_norm/denominator
            grad = grad + kl_grad + 0.001*delta/(safe_norm*denominator)
            total = nll+kl+penalty
            mx.eval(total, grad)
            if not mx.all(mx.isfinite(grad)).item():
                raise RuntimeError("Non-finite target gradient")
            record = dict(step=step, loss=total.item(), nll=nll.item(), kl=kl.item(),
                          delta_norm=mx.linalg.norm(delta).item(), grad_norm=mx.linalg.norm(grad).item(),
                          peak_metal_gb=mx.get_peak_memory()/1e9)
            history.append(record)
            if log:
                log(record)
            if step == steps-1 or (step % 2 == 0 and total.item() < 0.05):
                break
            t = step+1
            moment = 0.9*moment+0.1*grad
            variance = 0.999*variance+0.001*grad*grad
            delta = delta-lr*(moment/(1-0.9**t))/(mx.sqrt(variance/(1-0.999**t))+1e-8)
            if step % 2 == 0:
                delta = delta*mx.minimum(1.0, (32*norm)/mx.maximum(mx.linalg.norm(delta), 1e-12))
            mx.eval(delta, moment, variance)
            mx.clear_cache()
    finally:
        memory._probe_delta = None
    current = mx.stack(initials)
    # Match the native BF16 target addition before the FP32 allocation solve.
    targets = (current.astype(mx.bfloat16)+delta.astype(mx.bfloat16)).astype(mx.float32)
    return dict(current=np.array(current), targets=np.array(targets), key_sets=key_sets,
        history=history, wall_seconds=time.perf_counter()-started,
        expressions=sources, contexts=list(contexts), delta=np.array(delta))


def allocate_updates(targets, *, calibration_texts=(), tokenizer=None,
                     lambda_ridge=0.01, lambda_reuse=0.05):
    """Joint FP32 regularized allocation, with explicit local frequency calibration."""
    key_sets = [keys for target in targets for keys in target["key_sets"]]
    keys = list(dict.fromkeys((length, key) for row in key_sets for length, key in row.items()))
    if not keys:
        raise ValueError("No complete n-grams available for these subjects")
    columns = {key: i for i, key in enumerate(keys)}
    mapping = np.zeros((len(key_sets), len(keys)), dtype=np.float32)
    for row, key_set in enumerate(key_sets):
        for length, key in key_set.items():
            mapping[row, columns[(length, key)]] = 1
    counts = Counter()
    if tokenizer is not None:
        for text in calibration_texts:
            tokens = tokenizer.encode(text, add_special_tokens=False)
            for length in LENGTHS:
                for end in range(length, len(tokens)+1):
                    key = (length, tuple(tokens[end-length:end]))
                    if key in columns:
                        counts[key] += 1
    weights = []
    for length, key in keys:
        observed = [v for (n, _), v in counts.items() if n == length]
        count = counts[(length, key)]
        unique_counts = sorted(set(observed))
        percentile = unique_counts.index(count)/(len(unique_counts)-1) if count and len(unique_counts)>1 else 0
        frequency = min(1+9*percentile, 10)
        weights.append(min({2: 8., 3: 2., 4: 1.}[length]+frequency-1, 64))
    diagonal = lambda_ridge+lambda_reuse*np.array(weights, dtype=np.float32)
    current = np.concatenate([t["current"] for t in targets])
    expected = np.concatenate([t["targets"] for t in targets])
    system = mapping.T@mapping+np.diag(diagonal)
    updates = np.linalg.solve(system, mapping.T@(expected-current)).astype(np.float32)
    return dict(zip(keys, map(mx.array, updates))), dict(
        target_rows=len(mapping), keys=len(keys), condition_number=float(np.linalg.cond(system)),
        calibration_documents=len(calibration_texts),
        allocation_mse=float(np.mean((mapping@updates-(expected-current))**2)))
