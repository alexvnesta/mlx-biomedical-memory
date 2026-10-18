from __future__ import annotations

from datetime import datetime, timezone
import importlib.metadata
import json
import platform
from pathlib import Path
import random
import time
import hashlib

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
from mlx_lm import load
from huggingface_hub import snapshot_download

from .benchmark import LABELS, build, manifest, load_manifest
from .memory import MemoryModel, PhraseIndex, fingerprint

MODEL_ID = "Qwen/Qwen3-4B-MLX-bf16"
MODEL_REVISION = "94e68620a4b9db944b2f5b3c3633fbdf621d871b"


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def run(args):
    started = time.time()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "config.json").exists():
        raise ValueError("output already contains a run; choose a new output directory")
    mx.set_default_device(mx.gpu)
    mx.random.seed(args.seed)
    rng = random.Random(args.seed)
    config = vars(args).copy()
    code_digest = hashlib.sha256()
    for source in sorted(Path(__file__).parent.glob("*.py")):
        code_digest.update(source.name.encode())
        code_digest.update(source.read_bytes())
    config.update(
        model_revision=MODEL_REVISION,
        started_utc=datetime.now(timezone.utc).isoformat(),
        platform=platform.platform(),
        mlx=importlib.metadata.version("mlx"),
        mlx_lm=importlib.metadata.version("mlx-lm"),
        source_sha256=code_digest.hexdigest(),
        protocol="v0.2: closed-set A/B probing; supplied-edge composition; atomic premises fitted for every split",
    )
    write_json(out / "config.json", config)
    if args.dataset:
        data = load_manifest(args.dataset)
    elif args.arax_snapshot:
        from .biomedical import build_arax

        data = build_arax(args.arax_snapshot, args.seed)
        write_json(
            out / "arax-mapping.json",
            {"aliases": data["aliases"], "provenance": data["provenance"]},
        )
    else:
        data = build(
            args.seed,
            args.train_components,
            args.valid_components,
            args.test_components,
        )
    labels = data.get("labels", LABELS)
    write_json(out / "dataset.json", manifest(data))
    print("Loading model", flush=True)
    model_path = snapshot_download(
        args.model,
        revision=MODEL_REVISION,
        allow_patterns=[
            "*.json",
            "*.safetensors",
            "*.jinja",
            "merges.txt",
            "vocab.json",
        ],
    )
    backbone, tokenizer = load(model_path)
    all_keys = [key for x in data["all"] for key in x.keys]
    index = PhraseIndex(tokenizer, all_keys, args.slots)
    model = MemoryModel(
        backbone, args.slots, args.dimension, args.layer, args.memory_initialization
    )
    mx.eval(model.parameters())
    before = fingerprint(backbone)
    trainable = sum(x.size for _, x in tree_flatten(model.trainable_parameters()))
    if any(
        name.startswith("backbone.")
        for name, _ in tree_flatten(model.trainable_parameters())
    ):
        raise RuntimeError("backbone parameters unexpectedly trainable")
    choices = [
        tokenizer.encode(label, add_special_tokens=False) for label in ("A", "B")
    ]
    if any(len(tokens) != 1 for tokens in choices):
        raise ValueError(f"closed-set labels must be single tokens: {choices}")
    choice_ids = mx.array([tokens[0] for tokens in choices])
    encoded = {}

    def encode(example, oracle=False, paraphrase=False):
        key = (example.id, oracle, paraphrase)
        if key not in encoded:
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": example.prompt(oracle, paraphrase)}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            ids = tokenizer.encode(text, add_special_tokens=False)
            rows = index.encode(ids)
            if not oracle and sum(row != 0 for row in rows) != len(example.keys):
                raise ValueError(f"phrase addressing failed for {example.id}: {rows}")
            encoded[key] = (mx.array([ids]), mx.array([rows]))
        return encoded[key]

    # Verify coverage before an optimizer can run.
    for ex in data["all"]:
        encode(ex)
    write_json(out / "addresses.json", index.rows)

    def logits_for(example, mode, paraphrase=False):
        oracle = mode == "text_facts"
        ids, rows = encode(example, oracle, paraphrase)
        model.enabled = mode in {"memory", "shuffled_memory"}
        if mode == "shuffled_memory":
            rows = row_permutation[rows]
        return model(ids, rows)[:, choice_ids]

    # Permute only occupied rows, keeping the null row null and values on-distribution.
    row_permutation = list(range(args.slots))
    occupied = sorted(index.rows.values())
    shuffled = occupied[:]
    permutation_rng = random.Random(args.seed + 10000)
    if len(occupied) < 2:
        raise ValueError("shuffled control requires at least two memory entries")
    # Sattolo's single-cycle permutation is always a derangement, including odd N.
    for i in range(len(shuffled) - 1, 0, -1):
        j = permutation_rng.randrange(i)
        shuffled[i], shuffled[j] = shuffled[j], shuffled[i]
    for source, dest in zip(occupied, shuffled):
        row_permutation[source] = dest
    row_permutation = mx.array(row_permutation)

    def evaluate(examples, mode, paraphrase=False):
        predictions = []
        for ex in examples:
            logits = logits_for(ex, mode, paraphrase)
            probabilities = mx.softmax(logits.astype(mx.float32), axis=-1)[0]
            mx.eval(probabilities)
            prediction = labels[int(mx.argmax(probabilities).item())]
            predictions.append(
                {
                    "id": ex.id,
                    "kind": ex.kind,
                    "split": ex.split,
                    "expected": ex.label,
                    "prediction": prediction,
                    "correct": prediction == ex.label,
                    "probabilities": probabilities.tolist(),
                }
            )
        accuracy = sum(row["correct"] for row in predictions) / len(predictions)
        return {"n": len(predictions), "accuracy": accuracy, "predictions": predictions}

    # Read held-out labels only inside evaluation. No loss uses validation/test records.
    initial = {}
    for mode in ("base", "text_facts"):
        initial[mode] = evaluate(data["test"], mode)
        print(
            json.dumps(
                {
                    "stage": "baseline",
                    "mode": mode,
                    "accuracy": initial[mode]["accuracy"],
                }
            ),
            flush=True,
        )
        write_json(out / "baselines.json", initial)

    optimizer = optim.AdamW(learning_rate=args.learning_rate, weight_decay=0.0)

    def loss_fn(network, ids, rows, target):
        logits = network(ids, rows)[:, choice_ids].astype(mx.float32)
        return nn.losses.cross_entropy(logits, target, reduction="mean")

    vg = nn.value_and_grad(model, loss_fn)
    best_validation = -1.0
    log_path = out / "training.jsonl"
    atomic = [ex for ex in data["train"] if ex.kind == "atomic"]
    compositions = [ex for ex in data["train"] if ex.kind == "compose"]
    for step in range(1, args.steps + 1):
        # Alternate atomic and procedural objectives so recall isn't swamped by composition.
        pool = atomic if step % 2 else compositions
        ex = rng.choice(pool)
        length = encode(ex)[0].shape[1]
        # Same-length batches avoid introducing pad tokens or a new attention-mask implementation.
        compatible = [item for item in pool if encode(item)[0].shape[1] == length]
        batch = [ex] + rng.choices(compatible, k=args.batch_size - 1)
        ids = mx.concatenate([encode(item)[0] for item in batch], axis=0)
        rows = mx.concatenate([encode(item)[1] for item in batch], axis=0)
        model.enabled = True
        target = mx.array([labels.index(item.label) for item in batch])
        loss, grads = vg(model, ids, rows, target)
        optimizer.update(model, grads)
        mx.eval(model.memory.parameters(), optimizer.state, loss)
        record = {
            "step": step,
            "kind": ex.kind,
            "loss": float(loss.item()),
            "elapsed_seconds": time.time() - started,
        }
        with log_path.open("a") as f:
            f.write(json.dumps(record) + "\n")
        if step % 10 == 0 or step == 1:
            print(json.dumps(record), flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            valid = evaluate(data["valid"], "memory")
            print(
                json.dumps(
                    {"stage": "validation", "step": step, "accuracy": valid["accuracy"]}
                ),
                flush=True,
            )
            if valid["accuracy"] > best_validation:
                best_validation = valid["accuracy"]
                mx.save_safetensors(
                    str(out / "best-memory.safetensors"),
                    dict(tree_flatten(model.memory.parameters())),
                )
                write_json(
                    out / "selection.json",
                    {
                        "step": step,
                        "validation_accuracy": best_validation,
                        "selection_uses_test": False,
                    },
                )
        mx.clear_cache()
    mx.save_safetensors(
        str(out / "last-memory.safetensors"),
        dict(tree_flatten(model.memory.parameters())),
    )
    model.memory.load_weights(str(out / "best-memory.safetensors"))
    mx.eval(model.memory.parameters())
    results = {
        "initial": initial,
        "conditions": {},
        "atomic_recall": {},
        "protocol_audit": data["audit"],
        "trainable_parameters": trainable,
        "test_unique_questions": len({ex.keys for ex in data["test"]}),
        "test_components": len({ex.component for ex in data["test"]}),
        "scoring": "argmax over original LM logits for single-token A/B labels; not free-form generation",
    }
    test_keys = {key for ex in data["test"] for key in ex.keys}
    test_atomics = [ex for ex in atomic if ex.keys[0] in test_keys]
    for mode in ("base", "text_facts", "memory", "shuffled_memory"):
        results["conditions"][mode] = evaluate(data["test"], mode)
        results["atomic_recall"][mode] = evaluate(test_atomics, mode)
        write_json(out / "results.partial.json", results)
        print(
            json.dumps(
                {
                    "stage": "test",
                    "mode": mode,
                    "composition": results["conditions"][mode]["accuracy"],
                    "atomic_recall": results["atomic_recall"][mode]["accuracy"],
                }
            ),
            flush=True,
        )
    results["paraphrase_memory"] = evaluate(data["test"], "memory", paraphrase=True)
    if results["conditions"]["base"]["predictions"] != initial["base"]["predictions"]:
        raise RuntimeError("disabled-memory predictions changed during fitting")
    after = fingerprint(backbone)
    results["backbone"] = {
        "before_sha256": before,
        "after_sha256": after,
        "unchanged": before == after,
    }
    results["runtime"] = {
        "seconds": time.time() - started,
        "peak_memory_bytes": mx.get_peak_memory(),
    }
    results["interpretation"] = (
        "Pilot evidence only. Correct closed-set sign composition on supplied synthetic paths is not "
        "hidden-path discovery, novel biological knowledge, or clinical validation. A frozen backbone "
        "does not guarantee unchanged model behavior when its memory is enabled."
    )
    write_json(out / "results.json", results)
    if before != after:
        raise RuntimeError("frozen backbone changed")
    return results
