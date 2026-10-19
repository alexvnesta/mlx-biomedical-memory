"""Fixed 32-token timing of the selected readers, including reusable text KV prefixes."""
import copy
import hashlib
import json
from pathlib import Path
import random
import time

import mlx.core as mx
from mlx_lm import stream_generate
from mlx_lm.sample_utils import make_sampler
from engram_poc.decisive_model import Reader,load_reader,SPLIT_LAYER
from engram_poc.decisive_data import fields
from engram_poc.decisive_prompts import encode_prompt
from engram_poc.longcat_runtime import save,append,snapshot_source
from engram_poc.resource_guard import ResourceGuard


def main():
    out=Path('runs/decisive-latency');out.mkdir(exist_ok=False)
    corpus=json.loads(Path('data/processed/decisive-v1/dataset.json').read_text())
    cases=[c for split in ['iid','structure'] for c in corpus['splits'][split] if c['kind']=='reason']
    cases=[c for c in cases if int(c['id'].split('-')[-1])<6]
    assert len(cases)==12
    config=dict(source_sha256=snapshot_source(__file__,out/'source'),seed=1201,fixed_tokens=32,repeats=2,
        dataset_sha256=hashlib.sha256(Path('data/processed/decisive-v1/dataset.json').read_bytes()).hexdigest(),
        setup='One shared frozen model; adapter hot swaps excluded; priming costs separate.',
        scope='EOS suppressed fixed-length performance; no time-to-correct-answer claim.')
    save(out/'config.json',config);guard=ResourceGuard();model,tokenizer=load_reader('text');mx.eval(model.parameters())
    backbone=model.backbone;sampler=make_sampler(temp=0);priming=[];rows=[];checksums={}

    def swap(mode):
        nonlocal model
        # Strip only this experiment's LoRA/router wrappers and release the old codec.
        for layer in backbone.model.layers[SPLIT_LAYER:]:
            if hasattr(layer.mlp.router,'native'):layer.mlp.router=layer.mlp.router.native
            for attention in layer.self_attn:
                for name in ['q_a_proj','q_b_proj','kv_a_proj_with_mqa','o_proj']:
                    value=getattr(attention,name)
                    if hasattr(value,'linear'):setattr(attention,name,value.linear)
        model=Reader(backbone,mode)
        checkpoint=Path(f'runs/decisive-{mode}-1201')/('trained-selected.safetensors' if mode=='memory' else 'selected.safetensors')
        checksums[mode]=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        model.load_weights(str(checkpoint),strict=False);mx.eval(model.parameters());mx.clear_cache()

    def suppress_eos(tokens,logits):return logits.at[...,2].add(-1e9)

    # Alternate the outer method order across repetitions; use exactly one model allocation.
    for repeat in range(2):
        for mode in (['text','memory'] if repeat==0 else ['memory','text']):
            swap(mode);caches={}
            for case in cases:
                encoded=encode_prompt(tokenizer,case,mode);cut=encoded['prefix_cut'];cache=model.make_cache()
                start=time.perf_counter();f=mx.array(fields(case['records']));model.begin(f,encoded['placements'])
                value=model(mx.array([encoded['ids'][:cut]]),cache=cache);mx.eval(value)
                priming.append(dict(mode=mode,repeat=repeat,case_id=case['id'],tokens=cut,seconds=time.perf_counter()-start))
                caches[case['id']]=cache;del value;mx.clear_cache()
            schedule=[(case,cached) for case in cases for cached in [False,True]]
            random.Random(1201+repeat).shuffle(schedule)
            for case,cached in schedule:
                start=time.perf_counter();encoded=encode_prompt(tokenizer,case,mode);cut=encoded['prefix_cut'] if cached else 0
                cache=copy.deepcopy(caches[case['id']]) if cached else None
                model.begin(mx.array(fields(case['records'])),encoded['placements'],offset=cut)
                first=None;pieces=[];response=None
                for response in stream_generate(model,tokenizer,encoded['ids'][cut:],prompt_cache=cache,sampler=sampler,
                        max_tokens=32,logits_processors=[suppress_eos]):
                    if first is None:first=time.perf_counter()
                    pieces.append(response.text)
                end=time.perf_counter();assert response.generation_tokens==32
                row=dict(mode=mode,repeat=repeat,case_id=case['id'],prefix_cached=cached,
                    prompt_tokens=len(encoded['ids']),evaluated_prompt_tokens=len(encoded['ids'])-cut,
                    generated_tokens=32,wall_seconds=end-start,time_to_first_token_seconds=first-start,
                    decode_seconds=end-first,generation=''.join(pieces))
                rows.append(row);append(out/'predictions.jsonl',row);mx.clear_cache()
                if len(rows)%16==0:print('Measured',len(rows),guard.check(),flush=True)
            del caches;mx.clear_cache()
    save(out/'summary.json',dict(completed=True,rows=len(rows),priming=priming,checkpoint_sha256=checksums,resource=guard.check()))


if __name__=='__main__':main()
