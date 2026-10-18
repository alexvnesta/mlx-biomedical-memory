"""Bounded MLX replay of released EngramEdit edits on counterfactual MQuAKE.

Uses the upstream demonstrations, extraction, greedy decoding and token limits.
This is a quantized, serial, selected-case adaptation, not a full reproduction.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import random
import time

import mlx.core as mx
import numpy as np
from huggingface_hub import snapshot_download
from mlx_lm import stream_generate
from mlx_lm.sample_utils import make_sampler
from mlx_lm.models.cache import ArraysCache
from mlx_lm.tokenizer_utils import TokenizerWrapper
from mlx_lm.utils import load_model
from transformers import PreTrainedTokenizerFast

from engram_poc.longcat import MODEL_ID, MODEL_REVISION, install_memory
from engram_poc.longcat_keys import extract_answer, matches_answer


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, default=Path("data/processed/longcat"))
    parser.add_argument("--output", type=Path, default=Path("runs/longcat-mquake-smoke"))
    parser.add_argument("--cases", type=int, default=6)
    parser.add_argument("--questions-per-case", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--styles", nargs="+", choices=("standard", "cot"), default=["standard", "cot"])
    parser.add_argument("--memory-limit-gb", type=float, default=48)
    args = parser.parse_args()
    if not 1 <= args.cases <= 3000 or args.memory_limit_gb <= 0:
        parser.error("Choose a positive case count <= 3000 and positive memory limit")
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Choose an empty output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    mx.set_default_device(mx.gpu)
    mx.set_memory_limit(int(args.memory_limit_gb * 1e9))
    mx.set_cache_limit(256 * 1024**2)
    manifest_path = args.state / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for receipt in manifest["downloads"]:
        if digest(args.state / receipt["filename"]) != receipt["sha256"]:
            raise ValueError("Prepared input checksum failed")
    data = json.loads((args.state / "MQuAKE-CF-3k-v2.json").read_text())
    case_ids = list(range(len(data)))
    random.Random(0).shuffle(case_ids)
    case_ids = case_ids[:args.cases]
    processed_ids = set(manifest["processed_request_ids"])
    for case_id in case_ids:
        expected_ids = {f"{case_id}:{i}" for i in range(len(data[case_id]["requested_rewrite"]))}
        if not expected_ids.issubset(processed_ids):
            raise ValueError(f"Selected case {case_id} is not completely covered by released edits")
    code_hash = hashlib.sha256()
    for path in [Path(__file__), Path("engram_poc/longcat.py"), Path("engram_poc/longcat_keys.py")]:
        code_hash.update(path.read_bytes())
    config = dict(
        model=MODEL_ID, model_revision=MODEL_REVISION, quantization_bits=4,
        state=manifest["state_id"], state_revision=manifest["state_revision"],
        prepared_manifest_sha256=digest(manifest_path), source_sha256=code_hash.hexdigest(),
        case_ids=case_ids, selection="First cases after random.Random(0).shuffle; selected before generation",
        questions_per_case=args.questions_per_case, styles=args.styles,
        max_tokens=dict(standard=32, cot=128), temperature=0,
        mlx=importlib.metadata.version("mlx"), mlx_lm=importlib.metadata.version("mlx-lm"),
        platform=platform.platform(), started_utc=datetime.now(timezone.utc).isoformat(),
        memory_limit_gb=args.memory_limit_gb,
        limitations=["4-bit base versus paper BF16", "small selected subset", "serial rather than batched generation", "counterfactual edits are stipulated answers, not current-world facts", "text_edits supplies revised facts, not a retrieval system", "only one paraphrase per case by default", "no locality or general-capability evaluation"],
    )
    save(args.output / "config.json", config)
    print("Loading pinned 4-bit LongCat", flush=True)
    start = time.perf_counter()
    model_path = snapshot_download(
        MODEL_ID, revision=MODEL_REVISION, local_files_only=True,
        allow_patterns=["*.json", "*.jinja", "*.safetensors"],
    )
    # AutoTokenizer tries to load the custom PyTorch configuration for this
    # checkpoint. Its serialized tokenizer is sufficient; use it directly.
    model, model_config = load_model(Path(model_path))
    raw_tokenizer = PreTrainedTokenizerFast.from_pretrained(model_path, local_files_only=True)
    eos_ids = model_config["eos_token_id"]
    tokenizer = TokenizerWrapper(raw_tokenizer, eos_token_ids=[eos_ids] if isinstance(eos_ids, int) else eos_ids)
    mx.eval(model.parameters())
    load_seconds = time.perf_counter() - start
    print(f"Loaded in {load_seconds:.1f}s; Metal allocated {mx.get_active_memory()/1e9:.2f} GB", flush=True)
    # Verify the wrapper on the real model before trusting edited outputs.
    check = mx.array([[1, 174, 270, 314]])
    original = model(check)[:, -1, :]
    mx.eval(original)
    memory = install_memory(model, args.state)
    memory.enabled = False
    wrapped = model(check)[:, -1, :]
    mx.eval(wrapped)
    exact = mx.array_equal(original, wrapped).item()
    if not exact:
        raise RuntimeError("Disabled memory changed native model computation")
    del original, wrapped
    # Real edited keys exercise both the additive branch and suffix history.
    memory.enabled = True
    key = next(iter(memory._keys[4]))
    key_tokens = mx.array([list(key)])
    full_key = model(key_tokens)[:, -1, :]
    mx.eval(full_key)
    cache = model.make_cache()
    prefix = model(key_tokens[:, :2], cache=cache)
    mx.eval(prefix)
    cached_key = model(key_tokens[:, 2:], cache=cache)[:, -1, :]
    mx.eval(cached_key)
    cache_error = mx.max(mx.abs(full_key.astype(mx.float32) - cached_key.astype(mx.float32))).item()
    cache_top1_equal = mx.argmax(full_key, axis=-1).item() == mx.argmax(cached_key, axis=-1).item()
    if not cache_top1_equal or not np.isfinite(cache_error):
        raise RuntimeError("Cached edited forward failed the real-key sanity check")
    del full_key, cached_key, prefix, cache
    embedding_checks = {}
    for enabled in (False, True):
        memory.enabled = enabled
        whole = memory(key_tokens)
        emb_cache = ArraysCache(size=1)
        chunks = mx.concatenate([
            memory(key_tokens[:, :2], cache=emb_cache),
            memory(key_tokens[:, 2:], cache=emb_cache),
        ], axis=1)
        equal = mx.array_equal(whole, chunks).item()
        embedding_checks["edited" if enabled else "native"] = dict(
            full_chunked_exact=equal, dtype=str(whole.dtype))
        if not equal:
            raise RuntimeError("Native embedding or memory cache changed across chunks")
        del whole, chunks, emb_cache
    memory.enabled = False
    native_full = model(key_tokens)[:, -1, :]
    native_cache = model.make_cache()
    native_prefix = model(key_tokens[:, :2], cache=native_cache)
    mx.eval(native_prefix)
    native_cached = model(key_tokens[:, 2:], cache=native_cache)[:, -1, :]
    native_cache_error = mx.max(mx.abs(native_full.astype(mx.float32) - native_cached.astype(mx.float32))).item()
    del native_full, native_cached, native_prefix, native_cache
    mx.clear_cache()
    verification = dict(disabled_wrapper_exact=exact, load_seconds=load_seconds,
                        active_memory_gb=mx.get_active_memory()/1e9,
                        cached_edited_key_top1_equal=cache_top1_equal,
                        cached_edited_key_max_logit_difference=cache_error,
                        cached_native_key_max_logit_difference=native_cache_error,
                        embedding_cache_checks=embedding_checks,
                        edit_keys={k: len(v) for k, v in manifest["keys"].items()})
    save(args.output / "verification.json", verification)
    print(json.dumps(verification), flush=True)
    rows = []
    sampler = make_sampler(temp=0)
    for case_id in case_ids:
        record = data[case_id]
        target_answers = [record["new_answer"], *record.get("new_answer_alias", [])]
        old_answers = [record["answer"], *record.get("answer_alias", [])]
        updates_text = "\n".join(
            request["prompt"].format(request["subject"]) + " " + request["target_new"]["str"].lstrip()
            for request in record["requested_rewrite"]
        )
        for question_idx, question in enumerate(record["questions"][:args.questions_per_case]):
            for style in args.styles:
                cot = style == "cot"
                name = "multihop-cot-prompts.txt" if cot else "multihop-prompts.txt"
                demos = (args.state / name).read_text().rstrip()
                suffix = f"\n\nQuestion: {question}\nThoughts:" if cot else f"\nQ: {question} A:"
                for condition in ("base", "released_memory", "text_edits"):
                    memory.enabled = condition == "released_memory"
                    prompt = demos
                    if condition == "text_edits":
                        prompt += "\n\nFor this hypothetical question, use these revised facts:\n" + updates_text
                    prompt += suffix
                    # Match raw upstream prompting: no extra chat template or BOS.
                    tokens = tokenizer.encode(prompt, add_special_tokens=False)
                    started = time.perf_counter()
                    hit_counts_before = dict(memory._hit_counts)
                    chunks = []
                    response = None
                    for response in stream_generate(model, tokenizer, tokens,
                            max_tokens=128 if cot else 32, sampler=sampler):
                        chunks.append(response.text)
                    if response is None:
                        raise RuntimeError("Generation produced no response")
                    elapsed = time.perf_counter() - started
                    generation = "".join(chunks)
                    answer = extract_answer(generation, cot=cot)
                    row = dict(case_id=case_id, question_idx=question_idx, question=question,
                        hops=len(record["new_single_hops"]), style=style, condition=condition,
                        generation=generation, extracted_answer=answer,
                        target_answers=target_answers, original_answers=old_answers,
                        matches_updated=matches_answer(answer, target_answers),
                        matches_original=matches_answer(answer, old_answers),
                        prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                        prompt_tokens=len(tokens), generated_tokens=response.generation_tokens,
                        prompt_tps=response.prompt_tps, generation_tps=response.generation_tps,
                        wall_seconds=elapsed, peak_memory_gb=response.peak_memory)
                    row["edit_activations"] = {
                        str(length): memory._hit_counts[length] - hit_counts_before[length]
                        for length in memory._hit_counts
                    }
                    rows.append(row)
                    with (args.output / "predictions.jsonl").open("a") as stream:
                        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                    print(json.dumps({k: row[k] for k in ("case_id", "style", "condition", "extracted_answer", "matches_updated", "wall_seconds")}), flush=True)
                    mx.clear_cache()
    summary = {}
    for style in args.styles:
        summary[style] = {}
        for condition in ("base", "released_memory", "text_edits"):
            group = [r for r in rows if r["style"] == style and r["condition"] == condition]
            summary[style][condition] = dict(n=len(group),
                updated_answer_accuracy=sum(r["matches_updated"] for r in group)/len(group),
                original_answer_accuracy=sum(r["matches_original"] for r in group)/len(group),
                cases=len({r["case_id"] for r in group}),
                case_any_updated_accuracy=sum(
                    any(r["matches_updated"] for r in group if r["case_id"] == case_id)
                    for case_id in {r["case_id"] for r in group}
                ) / len({r["case_id"] for r in group}),
                median_wall_seconds=float(np.median([r["wall_seconds"] for r in group])))
    save(args.output / "summary.json", dict(conditions=summary,
         peak_metal_gb=mx.get_peak_memory()/1e9, verification=verification,
         interpretation="Exploratory quantized checkpoint replay; not biomedical reasoning or a full paper reproduction."))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
