"""Cache frozen causal prefix states for matched reader training only."""

import argparse
import hashlib
import json
from pathlib import Path
import time

import mlx.core as mx
from mlx.utils import tree_flatten

from engram_poc.decisive_model import load_reader,SPLIT_LAYER
from engram_poc.decisive_prompts import encode_prompt,target_text
from engram_poc.decisive_data import fields
from engram_poc.longcat import MODEL_ID,MODEL_REVISION
from engram_poc.longcat_runtime import snapshot_source,save,append
from engram_poc.resource_guard import ResourceGuard


def digest_frozen(model):
    digest=hashlib.sha256()
    for name,array in tree_flatten(model.parameters()):
        if name.startswith('memory.') or '.lora_a' in name or '.lora_b' in name:continue
        digest.update(name.encode());digest.update(str(array.shape).encode());digest.update(str(array.dtype).encode())
        raw=array.view(mx.uint8).reshape(-1)
        for offset in range(0,raw.size,32*1024**2):digest.update(memoryview(__import__('numpy').array(raw[offset:offset+32*1024**2])))
        mx.clear_cache()
    return digest.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=Path('data/processed/decisive-v1/dataset.json'))
    parser.add_argument('--output',type=Path,default=Path('runs/decisive-features-v1'))
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();out=args.output
    if out.exists() and not args.resume:raise ValueError('Preserve prior outputs')
    out.mkdir(exist_ok=True);(out/'features').mkdir(exist_ok=True)
    corpus=json.loads(args.data.read_text());data_sha=hashlib.sha256(args.data.read_bytes()).hexdigest()
    if not args.resume:
        save(out/'config.json',dict(model=MODEL_ID,revision=MODEL_REVISION,dataset_sha256=data_sha,
            source_sha256=snapshot_source(__file__,out/'source'),split_layer=SPLIT_LAYER,
            feature_semantics='Frozen causal teacher-forced prefix, train/dev only; no test labels encoded'))
        (out/'dataset.json').write_bytes(args.data.read_bytes())
    else:assert json.loads((out/'config.json').read_text())['dataset_sha256']==data_sha
    guard=ResourceGuard();print('Loading disk-backed frozen prefix',flush=True);model,tokenizer=load_reader('memory')
    print('Fingerprinting frozen base',flush=True);before=digest_frozen(model)
    rows=[]
    if (out/'manifest.jsonl').exists():rows=[json.loads(x) for x in (out/'manifest.jsonl').read_text().splitlines()]
    existing={(r['id'],r['mode']) for r in rows};parity=[];started=time.perf_counter()
    for split in ['dev','train']:
        for case in corpus['splits'][split]:
            for mode in ['text','memory']:
                if (case['id'],mode) in existing:continue
                encoded=encode_prompt(tokenizer,case,mode);target=tokenizer.encode(target_text(case),add_special_tokens=False)+[2]
                tokens=encoded['ids']+target[:-1];begin=time.perf_counter()
                h=model.prefix(mx.array([tokens]));mx.eval(h)
                if len(parity)<2:
                    f=mx.array(fields(case['records']));model.begin(f,encoded['placements'])
                    reference=model(mx.array([tokens]))[:,len(encoded['ids'])-1:]
                    cached=model.tail(h,f,encoded['placements'],loss_start=len(encoded['ids'])-1)
                    mx.eval(reference,cached)
                    max_error=mx.max(mx.abs(reference-cached)).item()
                    # Both paths project exactly the same target-position tensor.
                    assert max_error==0
                    parity.append(dict(mode=mode,case_id=case['id'],max_error=max_error))
                    del reference,cached
                path=out/'features'/(case['id']+'-'+mode+'.safetensors')
                mx.save_safetensors(str(path),{'hidden':h})
                row=dict(id=case['id'],split=split,kind=case['kind'],mode=mode,path=str(path),
                    feature_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),shape=list(h.shape),
                    token_sha256=hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),
                    target=target,loss_start=len(encoded['ids'])-1,positions=encoded['placements'],
                    fields=fields(case['records']),prefix_tokens=len(tokens),seconds=time.perf_counter()-begin)
                rows.append(row);append(out/'manifest.jsonl',row);del h;mx.clear_cache()
                if len(rows)%20==0:print('Cached '+str(len(rows))+' features '+json.dumps(guard.check()),flush=True)
    print('Final frozen-base fingerprint',flush=True);after=digest_frozen(model);assert before==after
    save(out/'summary.json',dict(completed=True,features=len(rows),seconds=time.perf_counter()-started,
        parity=parity,frozen_constants=dict(before=before,after=after,unchanged=True),resource=guard.check()))


if __name__=='__main__':main()
