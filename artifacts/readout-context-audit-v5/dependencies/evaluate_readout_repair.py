"""Freeze and causally test a readout reader; no reasoning is evaluated."""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import time

import mlx.core as mx
from mlx.utils import tree_flatten
from mlx_lm import stream_generate
from mlx_lm.sample_utils import make_sampler

from engram_poc.decisive_model import load_reader
from engram_poc.decisive_data import fields, atomic_solution
from engram_poc.resource_guard import ResourceGuard
from engram_poc.longcat_runtime import save, append, snapshot_source
from evaluate_decisive_reader import parse
from cache_decisive_features import digest_frozen


def module_at(path):
    spec=importlib.util.spec_from_file_location('frozen_readout_recipe',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    # The archived trainer may import data helpers. Bind the archived helpers,
    # rather than silently using a later checkout's generator or truth format.
    for filename,names in [('readout_data.py',['make_case','pair_split','noisy_prefix']),
                           ('decisive_data.py',['atomic_solution'])]:
        frozen=path.parent/filename
        if not frozen.exists():continue
        data_spec=importlib.util.spec_from_file_location('frozen_'+filename.removesuffix('.py'),frozen)
        data=importlib.util.module_from_spec(data_spec);data_spec.loader.exec_module(data)
        for name in names:
            if hasattr(data,name):setattr(module,name,getattr(data,name))
    return module


def test_cases(recipe):
    cases=[]
    for i in range(256):
        base=recipe.make_case(52000000+i,10,'test');base['id']=f'test-{i:04d}'
        cases.append(('normal',base,None))
        if i>=64:continue
        requested=next(r for r in base['records'] if r['id']==base['record_id'])
        for field in ['subject','object','sign','species','variant','source']:
            changed=copy.deepcopy(base);changed['id']+=f'-change-{field}'
            record=next(r for r in changed['records'] if r['id']==base['record_id'])
            if field=='sign':record[field]*=-1
            else:record[field]=(record[field]+1)%(16 if field in ['subject','object'] else 4 if field=='source' else 2)
            cases.append(('change_'+field,changed,base['id']))
        ordered=copy.deepcopy(base);ordered['id']+='-order'
        random.Random(53000000+i).shuffle(ordered['records']);cases.append(('order',ordered,base['id']))
        bound=copy.deepcopy(base);bound['id']+='-binding'
        for j,r in enumerate(bound['records']):
            for key in ['subject','object','sign','species','variant','source']:
                r[key]=base['records'][(j+1)%10][key]
        cases.append(('binding',bound,base['id']))
        masked=copy.deepcopy(base);masked['id']+='-masked';cases.append(('masked',masked,base['id']))
        wrong=copy.deepcopy(base);wrong['id']+='-wrong-bank'
        other=recipe.make_case(54000000+i,10,'test')
        wrong['injected_records']=[dict(r,**{k:v for k,v in s.items() if k!='id'}) for r,s in zip(base['records'],other['records'])]
        cases.append(('wrong_bank',wrong,base['id']))
    return cases


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--training',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--checkpoint-name',default='selected.safetensors')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--prepare-only',action='store_true')
    args=p.parse_args();out=args.output
    selection=json.loads((args.training/'selection.json').read_text())
    if selection['count']!=10 or selection['correct']/selection['total']<.96875:
        raise RuntimeError('Fresh test remains closed: first pass the ten-record generated development gate.')
    recipe=module_at(args.training/'source/train_readout_curriculum.py')
    checkpoint=args.training/args.checkpoint_name
    checkpoint_sha=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    cases=test_cases(recipe)
    if not args.resume:
        out.mkdir(exist_ok=False)
        # Freeze both test banks and checkpoint before any model outputs are opened.
        save(out/'cases.json',[dict(condition=c,case=case,parent_id=parent) for c,case,parent in cases])
        dependencies={name:hashlib.sha256((Path('scripts')/name).read_bytes()).hexdigest()
            for name in ['evaluate_decisive_reader.py','cache_decisive_features.py']}
        (out/'dependencies').mkdir()
        for name in dependencies:(out/'dependencies'/name).write_bytes((Path('scripts')/name).read_bytes())
        save(out/'config.json',dict(checkpoint_sha256=checkpoint_sha,training=str(args.training),
            checkpoint_name=args.checkpoint_name,source_sha256=snapshot_source(__file__,out/'source'),
            dependencies=dependencies,selection_sha256=hashlib.sha256((args.training/'selection.json').read_bytes()).hexdigest(),
            case_sha256=hashlib.sha256((out/'cases.json').read_bytes()).hexdigest(),
            test_seed_start=52000000,base_test_count=256,intervention_base_count=64,
            gate='normal exact >= 95%; causal controls reported separately',reasoning_evaluated=False))
        (out/'recipe.py').write_bytes((args.training/'source/train_readout_curriculum.py').read_bytes())
    else:
        config=json.loads((out/'config.json').read_text());assert checkpoint_sha==config['checkpoint_sha256']
        assert config['case_sha256']==hashlib.sha256((out/'cases.json').read_bytes()).hexdigest()
        assert json.loads((out/'cases.json').read_text())==[dict(condition=c,case=case,parent_id=parent) for c,case,parent in cases]
        assert config['selection_sha256']==hashlib.sha256((args.training/'selection.json').read_bytes()).hexdigest()
        source=hashlib.sha256()
        for path in [Path(__file__),*sorted(Path('engram_poc').glob('*.py'))]:source.update(path.name.encode());source.update(path.read_bytes())
        assert source.hexdigest()==config['source_sha256'],'Resume source changed'
        for name,digest in config['dependencies'].items():
            assert hashlib.sha256((Path('scripts')/name).read_bytes()).hexdigest()==digest
            assert hashlib.sha256((out/'dependencies'/name).read_bytes()).hexdigest()==digest
    if args.prepare_only:
        print('Frozen checkpoint and test banks; no model loaded or generations opened',flush=True);return
    guard=ResourceGuard();model,tok=load_reader('memory');model.load_weights(str(checkpoint),strict=False)
    mx.eval(model.parameters());loaded=dict(tree_flatten(model.trainable_parameters()));weights=mx.load(str(checkpoint))
    assert set(loaded)==set(weights) and all(mx.array_equal(loaded[k],v).item() for k,v in weights.items())
    before=digest_frozen(model)
    assert before=='dc059288f9bd691462722252054f8e9707f8dc700339c8731fca98f454839edb'
    rows=[json.loads(x) for x in (out/'predictions.jsonl').read_text().splitlines()] if (out/'predictions.jsonl').exists() else []
    done={r['case_id'] for r in rows};byid={case['id']:case for _,case,_ in cases}
    for condition,case,parent in cases:
        if case['id'] in done:continue
        e=recipe.encode(tok,case);injected=case.get('injected_records',case['records'])
        if parent and condition!='order':assert e['ids']==recipe.encode(tok,byid[parent])['ids']
        model.begin(mx.array(fields(injected)),e['positions'],enabled=condition!='masked')
        start=time.perf_counter();pieces=[]
        for response in stream_generate(model,tok,e['ids'],sampler=make_sampler(temp=0),max_tokens=100):
            pieces.append(response.text)
            if parse(''.join(pieces),'atomic') is not None:break
        text=''.join(pieces);parsed=parse(text,'atomic');expected=atomic_solution(case)
        injected_expected=atomic_solution(dict(case,records=injected))
        row=dict(case_id=case['id'],condition=condition,parent_id=parent,records=case['records'],
            injected_records=injected,record_id=case['record_id'],prompt=e['prompt'],prompt_tokens=len(e['ids']),
            generation=text,parsed=parsed,expected=expected,correct=parsed==expected,
            injected_expected=injected_expected,injected_correct=parsed==injected_expected,
            field_correct={k:(parsed or {}).get(k)==v for k,v in expected.items()},
            seconds=time.perf_counter()-start)
        append(out/'predictions.jsonl',row);rows.append(row);mx.clear_cache()
        if len(rows)%32==0:print(json.dumps(dict(evaluated=len(rows),resource=guard.check())),flush=True)
    after=digest_frozen(model);assert after==before
    conditions={}
    for condition in sorted({r['condition'] for r in rows}):
        subset=[r for r in rows if r['condition']==condition]
        conditions[condition]=dict(correct=sum(r['correct'] for r in subset),total=len(subset),
            injected_correct=sum(r['injected_correct'] for r in subset),
            field_correct={k:sum(r['field_correct'][k] for r in subset) for k in rows[0]['expected']})
    save(out/'summary.json',dict(completed=True,rows=len(rows),conditions=conditions,
        gate_passed=conditions['normal']['correct']/conditions['normal']['total']>=.95,
        frozen_constants=dict(before=before,after=after),checkpoint_load_exact=True,resource=guard.check()))


if __name__=='__main__':main()
