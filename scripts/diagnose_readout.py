"""Audit the failed reader without fitting or using new evaluation cases."""
import json
from pathlib import Path
import time

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from mlx_lm import stream_generate
from mlx_lm.sample_utils import make_sampler

from engram_poc.decisive_model import load_reader
from engram_poc.decisive_data import fields, atomic_solution
from engram_poc.decisive_prompts import encode_prompt, target_text
from engram_poc.resource_guard import ResourceGuard
from engram_poc.longcat_runtime import save, append, snapshot_source
from evaluate_decisive_reader import parse


def main():
    out=Path('runs/readout-diagnosis-v1');out.mkdir(exist_ok=False)
    guard=ResourceGuard();snapshot_source(__file__,out/'source')
    model,tok=load_reader('memory');mx.eval(model.parameters())
    training=Path('runs/decisive-memory-1202')
    full=mx.load(str(training/'last.safetensors'))
    model.load_weights(list(full.items()),strict=False)
    loaded=dict(tree_flatten(model.trainable_parameters()))
    assert set(loaded)==set(full)
    assert all(mx.array_equal(loaded[k],v).item() for k,v in full.items())
    corpus=json.loads(Path('runs/decisive-features-v1/dataset.json').read_text())
    cases=[c for split in ['train','dev'] for c in corpus['splits'][split] if c['kind']=='atomic']
    cases=cases[:4]+cases[-4:]
    manifest=[json.loads(x) for x in Path('runs/decisive-features-v1/manifest.jsonl').read_text().splitlines()]
    byid={r['id']:r for r in manifest if r['mode']=='memory'}
    for precision in ['fp32','bf16']:
        weights=full if precision=='fp32' else {k:v.astype(mx.bfloat16) for k,v in full.items()}
        model.load_weights(list(weights.items()),strict=False)
        mx.eval(model.trainable_parameters())
        for index,case in enumerate(cases):
            e=encode_prompt(tok,case,'memory');f=mx.array(fields(case['records']))
            target=tok.encode(target_text(case),add_special_tokens=False)+[2]
            model.begin(f,e['placements'])
            tokens=mx.array([e['ids']+target[:-1]])
            row=byid[case['id']];h=mx.load(row['path'])['hidden']
            cached=model.tail(h,f,e['placements'],loss_start=len(e['ids'])-1)
            direct=model(tokens)[:,len(e['ids'])-1:]
            err=mx.max(mx.abs(cached-direct)).item()
            ce=nn.losses.cross_entropy(direct.astype(mx.float32),mx.array([target]),reduction='none')[0]
            mx.eval(ce)
            teacher=[tok.decode([int(x)]) for x in mx.argmax(direct,axis=-1)[0].tolist()]
            # Causal KV incremental logits must agree with uncached logits.
            cache=model.make_cache();model.begin(f,e['placements'])
            pref=model(mx.array([e['ids']]),cache=cache)[:,-1:]
            incremental_error=mx.max(mx.abs(pref-direct[:,:1])).item()
            model.begin(f,e['placements']);pieces=[];start=time.perf_counter()
            for response in stream_generate(model,tok,e['ids'],sampler=make_sampler(temp=0),max_tokens=140):
                pieces.append(response.text)
                if parse(''.join(pieces),'atomic') is not None:break
            parsed=parse(''.join(pieces),'atomic');expected=atomic_solution(case)
            result=dict(precision=precision,case_id=case['id'],cached_direct_max_error=err,
                prefix_incremental_max_error=incremental_error,nll=ce.tolist(),target_tokens=[tok.decode([v]) for v in target],
                teacher_argmax=teacher,parsed=parsed,expected=expected,correct=parsed==expected,
                field_correct={k:(parsed or {}).get(k)==v for k,v in expected.items()},
                generation=''.join(pieces),seconds=time.perf_counter()-start)
            append(out/'predictions.jsonl',result);print(json.dumps(result),flush=True)
            del cache,pref,direct,cached,h,tokens;mx.clear_cache();guard.check()
    save(out/'summary.json',dict(completed=True,checkpoint_load_exact=True,resource=guard.check()))


if __name__=='__main__':main()
