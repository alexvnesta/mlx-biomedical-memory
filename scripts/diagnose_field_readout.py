"""Separate field decoding from autoregressive whole-record decoding on development only."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import mlx.core as mx
from mlx_lm import stream_generate
from mlx_lm.sample_utils import make_sampler
from engram_poc.decisive_model import load_reader
from engram_poc.decisive_data import fields
from engram_poc.resource_guard import ResourceGuard
from engram_poc.longcat_runtime import save,append,snapshot_source
from evaluate_decisive_reader import parse
from evaluate_readout_repair import module_at


def main():
    p=argparse.ArgumentParser();p.add_argument('--training',type=Path,required=True)
    p.add_argument('--checkpoint-name',default='step-0512.safetensors');p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();out=args.output;out.mkdir(exist_ok=False)
    recipe=module_at(args.training/'source/train_readout_curriculum.py');ckpt=args.training/args.checkpoint_name
    save(out/'config.json',dict(source_sha256=snapshot_source(__file__,out/'source'),
        checkpoint_sha256=hashlib.sha256(ckpt.read_bytes()).hexdigest(),training=str(args.training),checkpoint_name=args.checkpoint_name,
        purpose='Development diagnostic only; no test cases'))
    guard=ResourceGuard();model,tok=load_reader('memory');model.load_weights(str(ckpt),strict=False);mx.eval(model.parameters())
    rows=[];conditions=['full','subject','object','sign','species','variant','source']
    for i in range(32):
        for condition in conditions:
            case=recipe.make_case(41010000+i,1,'dev')
            if condition!='full':case['field']=condition
            e=recipe.encode(tok,case);model.begin(mx.array(fields(case['records'])),e['positions']);pieces=[]
            for response in stream_generate(model,tok,e['ids'],sampler=make_sampler(temp=0),max_tokens=100):
                pieces.append(response.text)
                if parse(''.join(pieces),'atomic') is not None:break
            text=''.join(pieces);obj=parse(text,'atomic');expected=recipe.expected(case)
            row=dict(index=i,condition=condition,case=case,expected=expected,parsed=obj,generation=text,correct=obj==expected)
            append(out/'predictions.jsonl',row);rows.append(row);mx.clear_cache()
        if i%8==7:print(json.dumps(dict(banks=i+1,resource=guard.check())),flush=True)
    save(out/'summary.json',dict(completed=True,counts={c:sum(r['correct'] for r in rows if r['condition']==c) for c in conditions},
        per_condition=32,all_six_fields_correct=sum(all(r['correct'] for r in rows if r['index']==i and r['condition']!='full') for i in range(32)),resource=guard.check()))


if __name__=='__main__':main()
