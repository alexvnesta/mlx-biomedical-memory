"""Fixed-output latency comparison; no accuracy or clinical speed claim implied."""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import time

from huggingface_hub import snapshot_download
import mlx.core as mx
from mlx_lm import stream_generate
from mlx_lm.sample_utils import make_sampler
from mlx_lm.tokenizer_utils import TokenizerWrapper
from mlx_lm.utils import load_model
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizerFast

from engram_poc.longcat import MODEL_ID, MODEL_REVISION, install_memory
from engram_poc.retrieval import Retriever


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=Path('runs/comparison-v1'))
    parser.add_argument('--data', type=Path, default=Path('data/processed/comparison-v1'))
    parser.add_argument('--output', type=Path, default=Path('runs/comparison-latency'))
    parser.add_argument('--repeats', type=int, default=2)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Choose a new output directory')
    args.output.mkdir(parents=True)
    dataset = json.loads((args.data/'dataset.json').read_text())
    receipt = json.loads((args.data/'embedding-receipt.json').read_text())
    assert hashlib.sha256((args.data/'dataset.json').read_bytes()).hexdigest() == receipt['corpus_sha256']
    complete = json.loads((args.run/'summary.json').read_text())
    assert complete['completed'] and complete['frozen_constants']['unchanged']
    config = dict(model=MODEL_ID, revision=MODEL_REVISION, fixed_tokens=32, repeats=args.repeats,
        seed=601, dataset_sha256=receipt['corpus_sha256'],
        memory_sha256=json.loads((args.run/'memory/manifest.json').read_text())['updates_sha256'],
        cache='warm prefix versus no reused prefix; model and OS caches are warm in both',
        scoring='EOS suppressed, fixed length, no quality or time-to-correct-answer inference')
    (args.output/'source').mkdir()
    source_digest=hashlib.sha256()
    for source in [Path(__file__),*sorted(Path('engram_poc').glob('*.py'))]:
        source_digest.update(source.name.encode());source_digest.update(source.read_bytes())
        (args.output/'source'/source.name).write_bytes(source.read_bytes())
    config['source_sha256']=source_digest.hexdigest()
    (args.output/'config.json').write_text(json.dumps(config, indent=2)+'\n')
    (args.output/'source.py').write_bytes(Path(__file__).read_bytes())
    mx.set_memory_limit(48_000_000_000); mx.set_cache_limit(128*1024**2)
    model_path = snapshot_download(MODEL_ID, revision=MODEL_REVISION, local_files_only=True,
        allow_patterns=['*.json', '*.jinja', '*.safetensors'])
    model, _ = load_model(Path(model_path))
    tokenizer = TokenizerWrapper(PreTrainedTokenizerFast.from_pretrained(model_path, local_files_only=True), eos_token_ids=[2])
    memory = install_memory(model, args.run/'memory')
    torch.set_num_threads(6)
    embed_tok = AutoTokenizer.from_pretrained(receipt['model'], revision=receipt['revision'], local_files_only=True)
    embed_model = AutoModel.from_pretrained(receipt['model'], revision=receipt['revision'], local_files_only=True).eval().to('cpu')
    retriever = Retriever(dataset, np.load(args.data/'embeddings.npz')['documents'], {})
    sampler = make_sampler(temp=0)
    selected = [q for q in dataset['questions'] if q['kind']=='composition' and q['id'].endswith('-0')]
    instruction = ('Use the frozen reference, which can be stored internally or supplied below. '
        'Its fictional or anonymized records override ordinary world knowledge. '
        'If the reference has no matching record, answer UNKNOWN. '
        'This is a record-lookup task, not a treatment claim. '
        'Return only the precise answer label, without explanation.\n')
    caches = {}; priming = []; rows = []

    def suppress_eos(tokens, logits):
        return logits.at[..., 2].add(-1e9)

    def evaluate(q, method, cached, warmup=False):
        start = time.perf_counter(); memory.enabled = method == 'memory'
        rt = time.perf_counter()
        if method == 'hybrid':
            encoded = embed_tok(['Represent this sentence for searching relevant passages: '+q['text']],
                truncation=True, max_length=256, return_tensors='pt')
            with torch.no_grad():
                vec = embed_model(**encoded).last_hidden_state[:, 0]
                retriever.query_vectors[q['id']] = torch.nn.functional.normalize(vec, p=2, dim=1).numpy()[0]
        docs = retriever.retrieve(q, method) if method != 'memory' else []
        retrieval = time.perf_counter()-rt
        prefix = instruction
        if method != 'memory':
            prefix += 'Reference records:\n'+'\n'.join(d['text'] for d in docs)+'\n'
        prompt = prefix+'Question: '+q['text']+'\nAnswer:'
        tokens = tokenizer.encode(prompt, add_special_tokens=False)
        remaining = tokens; cache = None
        if cached:
            cut = max(0, len(tokenizer.encode(prefix, add_special_tokens=False))-3)
            key = (method, tuple(tokens[:cut]))
            if key not in caches:
                t = time.perf_counter(); cache = model.make_cache()
                hidden = model.model(mx.array([tokens[:cut]]), cache=cache)
                mx.eval(hidden)
                priming.append(dict(method=method, tokens=cut, seconds=time.perf_counter()-t))
                caches[key] = cache
            cache = copy.deepcopy(caches[key]); remaining = tokens[cut:]
        gs = time.perf_counter(); first = None; text = []; response = None
        for response in stream_generate(model, tokenizer, remaining, prompt_cache=cache, sampler=sampler,
                max_tokens=32, logits_processors=[suppress_eos]):
            if first is None: first = time.perf_counter()
            text.append(response.text)
        end = time.perf_counter()
        assert response is not None and response.generation_tokens == 32, response
        row = dict(question_id=q['id'], method=method, prefix_cached=cached, prompt_tokens=len(tokens),
            evaluated_prompt_tokens=len(remaining), generated_tokens=response.generation_tokens,
            retrieval_seconds=retrieval, generation_seconds=end-gs, wall_seconds=end-start,
            time_to_first_token_seconds=first-start, decode_seconds=end-first, generation=''.join(text))
        if not warmup:
            rows.append(row)
            with (args.output/'predictions.jsonl').open('a') as stream: stream.write(json.dumps(row)+'\n')
            print(json.dumps({k:v for k,v in row.items() if k!='generation'}), flush=True)
        mx.clear_cache()

    # Prime every measured prefix before starting balanced measurements. These
    # setup costs remain separately visible and are never silently amortized.
    for q in selected:
        for method in ['memory', 'kg', 'hybrid']:
            evaluate(q, method, True, warmup=True)
    randomizer = random.Random(601)
    for repeat in range(args.repeats):
        queries = list(selected); randomizer.shuffle(queries)
        for q in queries:
            conditions = [(m, c) for m in ['memory','kg','hybrid'] for c in [False,True]]
            randomizer.shuffle(conditions)
            for method, cached in conditions: evaluate(q, method, cached)
    summary = dict(completed=True, measurements=len(rows), queries=len(selected), priming=priming,
        peak_metal_gb=mx.get_peak_memory()/1e9, config=config)
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print('COMPLETED', flush=True)


if __name__ == '__main__': main()
