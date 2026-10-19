"""Generate and independently verify unseen-world relationship proofs."""

import argparse
import copy
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import random
import time

import mlx.core as mx
from mlx_lm import stream_generate
from mlx_lm.sample_utils import make_sampler

from engram_poc.decisive_data import fields,grade,solution,atomic_solution
from engram_poc.decisive_model import load_reader
from engram_poc.decisive_prompts import encode_prompt
from engram_poc.longcat_runtime import snapshot_source,save,append
from engram_poc.resource_guard import ResourceGuard
from cache_decisive_features import digest_frozen


def parse(text,kind):
    decoder=json.JSONDecoder()
    for i,char in enumerate(text):
        if char!='{':continue
        try:obj,_=decoder.raw_decode(text[i:])
        except json.JSONDecodeError:continue
        if isinstance(obj,dict) and ('record' in obj if kind=='atomic' else 'answer' in obj):return obj
    return None


def variations(corpus):
    result=[]
    for split in ['iid','structure']:
        for case in corpus['splits'][split]:result.append((split,case,'normal'))
    for case in corpus['interventions']:result.append(('intervention',case,'normal'))
    for split in ['iid','structure']:
        cases=[c for c in corpus['splits'][split] if c['kind']=='reason']
        for case in cases[:32]:
            changed=copy.deepcopy(case);rng=random.Random(7000+int(case['id'].split('-')[-1]))
            identifiers=[r['id'] for r in changed['records']];rng.shuffle(identifiers)
            for record,new_id in zip(changed['records'],identifiers):record['id']=new_id
            changed.update(id=case['id']+'-binding',parent_id=case['id'])
            result.append(('binding',changed,'normal'))
            result.append(('masked',case,'masked'))
        for case in cases[:16]:
            changed=copy.deepcopy(case);random.Random(9000+int(case['id'].split('-')[-1])).shuffle(changed['records'])
            changed.update(id=case['id']+'-order',parent_id=case['id']);result.append(('order',changed,'normal'))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training',type=Path,required=True)
    parser.add_argument('--features',type=Path,default=Path('runs/decisive-features-v1'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--development',action='store_true')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();out=args.output;trained=json.loads((args.training/'config.json').read_text())
    assert json.loads((args.training/'summary.json').read_text())['completed']
    corpus=json.loads((args.features/'dataset.json').read_text());mode=trained['mode']
    if out.exists() and not args.resume:raise ValueError('Preserve prior evaluations')
    out.mkdir(exist_ok=True)
    checkpoint=args.training/'best.safetensors';checkpoint_sha=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if not args.resume:
        config=dict(mode=mode,seed=trained['seed'],started_utc=datetime.now(timezone.utc).isoformat(),
            source_sha256=snapshot_source(__file__,out/'source'),dataset_sha256=trained['dataset_sha256'],
            checkpoint_sha256=checkpoint_sha,max_tokens=256,development=args.development,
            dependency_sha256=hashlib.sha256(Path('scripts/cache_decisive_features.py').read_bytes()).hexdigest())
        save(out/'config.json',config);(out/'cache_decisive_features.py').write_bytes(Path('scripts/cache_decisive_features.py').read_bytes())
    else:
        config=json.loads((out/'config.json').read_text());assert checkpoint_sha==config['checkpoint_sha256']
        source=hashlib.sha256()
        for path in [Path(__file__),*sorted(Path('engram_poc').glob('*.py'))]:source.update(path.name.encode());source.update(path.read_bytes())
        assert source.hexdigest()==config['source_sha256'],'Resume source changed'
    guard=ResourceGuard();print('Loading full '+mode+' reader',flush=True);model,tokenizer=load_reader(mode)
    model.load_weights(str(checkpoint),strict=False);mx.eval(model.parameters())
    before=digest_frozen(model);sampler=make_sampler(temp=0);rows=[]
    if (out/'predictions.jsonl').exists():rows=[json.loads(x) for x in (out/'predictions.jsonl').read_text().splitlines()]
    existing={(r['stage'],r['case_id']) for r in rows}
    cases=[('development',c,'normal') for c in corpus['splits']['dev']] if args.development else variations(corpus)
    if mode=='text':cases=[item for item in cases if item[0]!='masked']
    for stage,case,condition in cases:
        if (stage,case['id']) in existing:continue
        started=time.perf_counter();encoded=encode_prompt(tokenizer,case,mode)
        model.begin(mx.array(fields(case['records'])),encoded['placements'],enabled=condition!='masked')
        pieces=[];parsed=None;response=None;first=None
        for response in stream_generate(model,tokenizer,encoded['ids'],sampler=sampler,max_tokens=256):
            if first is None:first=time.perf_counter()
            pieces.append(response.text);parsed=parse(''.join(pieces),case['kind'])
            if parsed is not None:break
        text=''.join(pieces)
        row=dict(stage=stage,case_id=case['id'],namespace=case['namespace'],kind=case['kind'],depth=case['depth'],
            parent_id=case.get('parent_id'),mode=mode,seed=trained['seed'],condition=condition,
            query=case['query'],records=case['records'],record_id=case.get('record_id'),
            expected=atomic_solution(case) if case['kind']=='atomic' else solution(case),parsed=parsed,
            generation=text,prompt=encoded['prompt'],prompt_tokens=len(encoded['ids']),
            generated_tokens=response.generation_tokens if response else 0,
            wall_seconds=time.perf_counter()-started,
            time_to_first_token_seconds=first-started if first else None,**grade(case,parsed))
        rows.append(row);append(out/'predictions.jsonl',row);mx.clear_cache()
        if len(rows)%16==0:print('Evaluated '+str(len(rows))+' '+json.dumps(guard.check()),flush=True)
    after=digest_frozen(model);assert before==after
    summary=dict(completed=True,rows=len(rows),resource=guard.check(),
        frozen_constants=dict(before=before,after=after,unchanged=True))
    save(out/'summary.json',summary);print('COMPLETED '+json.dumps(summary),flush=True)


if __name__=='__main__':main()
