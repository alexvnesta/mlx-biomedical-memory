"""Fit independent tuple values and rule motifs before richer held-out tests."""

import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import random
import time

import mlx.core as mx
from mlx.utils import tree_flatten
from mlx_lm import stream_generate
from mlx_lm.sample_utils import make_sampler
import numpy as np

from engram_poc.longcat import MODEL_ID,MODEL_REVISION
from engram_poc.longcat_editor import fit_target,allocate_updates
from engram_poc.longcat_runtime import load_native,frozen_digest,snapshot_source,save,append
from engram_poc.scoped_memory import install_scoped
from engram_poc.relational import parse_response,grade
from engram_poc.relational_prompts import direct_prompt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=Path('data/processed/relational-v1/dataset.json'))
    parser.add_argument('--output',type=Path,default=Path('runs/relational-fit-v1'))
    args=parser.parse_args();out=args.output
    if out.exists():raise ValueError('Preserve existing outcomes; choose a new output')
    out.mkdir(parents=True)
    corpus=json.loads(args.data.read_text());(out/'dataset.json').write_bytes(args.data.read_bytes())
    config=dict(model=MODEL_ID,revision=MODEL_REVISION,started_utc=datetime.now(timezone.utc).isoformat(),
        dataset_sha256=hashlib.sha256(args.data.read_bytes()).hexdigest(),
        source_sha256=snapshot_source(__file__,out/'source'),steps=25,lr=.5,
        contexts=['{}','Reference lookup:\n{}'],supervision='Atomic tuple values and six rule definitions only; no composed answers or test proofs')
    save(out/'config.json',config)
    print('Loading native LongCat',flush=True);model,tokenizer=load_native()
    ids=mx.array([[1,121,414,390]]);native=model(ids);mx.eval(native)
    memory=install_scoped(model);memory.begin('native');empty=model(ids);mx.eval(empty)
    assert mx.array_equal(native,empty).item(),'Native forward changed'
    assert not tree_flatten(model.trainable_parameters())
    del native,empty
    print('Fingerprinting frozen weights',flush=True);before=frozen_digest(model)
    motifs={r['id']:r for r in corpus['motifs']};records={r['id']:r for r in corpus['records']}
    sampler=make_sampler(temp=0);dev=[]
    for case in corpus['dev_cases']:
        prompt=direct_prompt(case,motifs[case['motif_id']],[records[i] for i in case['record_ids']],True)
        pieces=[];obj=None;memory.begin('native')
        for response in stream_generate(model,tokenizer,tokenizer.encode(prompt,add_special_tokens=False),max_tokens=160,sampler=sampler):
            pieces.append(response.text);obj=parse_response(''.join(pieces))
            if obj and 'answer' in obj:break
        obj=obj or {};evidence=obj.get('evidence',[])
        if not isinstance(evidence,list) or not all(isinstance(x,str) for x in evidence):evidence=[]
        row=dict(case_id=case['id'],generation=''.join(pieces),parsed=obj,
            **grade(case,str(obj.get('answer','INVALID')),evidence,case['record_ids']))
        dev.append(row);append(out/'dev-gold.jsonl',row);print('Dev '+json.dumps(row),flush=True)
    if not any('answer' in row['parsed'] for row in dev):raise RuntimeError('Development JSON interface failed; fitting not started')
    requests=corpus['atomic_requests']+corpus['motifs'];results=[];(out/'targets').mkdir()
    start=time.perf_counter()
    for index,request in enumerate(requests):
        print(f"Fitting {index+1}/{len(requests)}: {request['id']}",flush=True);memory.begin('native')
        def log(row):
            append(out/'training.jsonl',dict(request_id=request['id'],**row))
            if row['step']%5==0:print(json.dumps(dict(request_id=request['id'],**row)),flush=True)
        result=fit_target(model,memory,tokenizer,request,steps=25,contexts=config['contexts'],
            expressions=request['expressions'],log=log)
        results.append(result)
        stem=out/'targets'/request['id']
        np.savez(stem.with_suffix('.npz'),current=result['current'],targets=result['targets'],delta=result['delta'])
        save(stem.with_suffix('.json'),{k:v for k,v in result.items() if k not in ['current','targets','delta']})
    updates,allocation=allocate_updates(results,calibration_texts=[r['prompt'].format(r['subject'])+' '+r['target'] for r in requests],tokenizer=tokenizer)
    memory.add_updates(updates)
    permutation=list(range(1,len(requests)+1));rng=random.Random(701)
    for group in [list(range(1,len(corpus['atomic_requests'])+1)),list(range(len(corpus['atomic_requests'])+1,len(requests)+1))]:
        rng.shuffle(group)
        for a,b in zip(group,group[1:]+group[:1]):permutation[a-1]=b
    memory.store([r['key'] for r in requests],[mx.array(r['delta']) for r in results],permutation)
    memory.save_scoped(out/'memory',dict(config=config,allocation=allocation))
    seconds=time.perf_counter()-start;print('Fingerprinting after fitting',flush=True);after=frozen_digest(model)
    assert before==after,'Frozen weights changed'
    summary=dict(completed=True,requests=len(requests),fit_seconds=seconds,allocation=allocation,
        dev_gold_correct=sum(r['correct'] for r in dev),dev_gold_proof_valid=sum(r['proof_valid'] for r in dev),
        frozen_constants=dict(before=before,after=after,unchanged=before==after),peak_metal_gb=mx.get_peak_memory()/1e9)
    save(out/'summary.json',summary);print('COMPLETED '+json.dumps(summary),flush=True)


if __name__=='__main__':main()
