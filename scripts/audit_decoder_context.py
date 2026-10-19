"""Development-only intervention on previous output values, with fixed memory."""
import argparse
import hashlib
import json
from pathlib import Path
import re

import mlx.core as mx
import mlx.nn as nn

from engram_poc.decisive_model import load_reader
from engram_poc.decisive_data import fields,atomic_solution
from engram_poc.resource_guard import ResourceGuard
from engram_poc.longcat_runtime import save,append,snapshot_source
from evaluate_readout_repair import module_at


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--before',type=Path,default=Path('runs/readout-curriculum-v4/step-0512.safetensors'))
    p.add_argument('--after',type=Path,required=True)
    p.add_argument('--training',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();out=args.output;out.mkdir(exist_ok=False)
    recipe=module_at(args.training/'source/train_readout_curriculum.py')
    checkpoints=dict(before=args.before,after=args.after)
    save(out/'config.json',dict(source_sha256=snapshot_source(__file__,out/'source'),
        checkpoints={k:dict(path=str(v),sha256=hashlib.sha256(v.read_bytes()).hexdigest()) for k,v in checkpoints.items()},
        semantics='Teacher-forced object token accuracy/NLL under interventions on the previously emitted subject; fixed memory; development only'))
    guard=ResourceGuard();model,tok=load_reader('memory');mx.eval(model.parameters());rows=[]
    for phase,checkpoint in checkpoints.items():
        model.load_weights(str(checkpoint),strict=False);mx.eval(model.trainable_parameters())
        for index in range(32):
            case=recipe.make_case(41010000+index,1,'dev');encoded=recipe.encode(tok,case)
            truth=atomic_solution(case)
            for previous_subject in range(16):
                context=dict(truth,subject=f'N{previous_subject}')
                text=json.dumps(context,separators=(',',':'))
                target=tok.encode(text,add_special_tokens=False)+[2]
                offsets=tok._tokenizer(text,add_special_tokens=False,return_offsets_mapping=True)['offset_mapping']
                span=re.search(r'"object":"([^\"]+)"',text).span(1)
                positions=[i for i,(a,b) in enumerate(offsets) if a<span[1] and b>span[0]]
                model.begin(mx.array(fields(case['records'])),encoded['positions'])
                logits=model(mx.array([encoded['ids']+target[:-1]]))[0]
                logits=logits[mx.array([len(encoded['ids'])-1+i for i in positions])].astype(mx.float32)
                expected=[target[i] for i in positions];predicted=mx.argmax(logits,axis=-1).tolist()
                nll=nn.losses.cross_entropy(logits,mx.array(expected),reduction='mean').item()
                row=dict(phase=phase,index=index,record=truth,previous_subject=f'N{previous_subject}',
                    true_previous_subject=previous_subject==int(truth['subject'][1:]),
                    teacher_object_tokens=expected,argmax_tokens=predicted,object_token_exact=predicted==expected,nll=nll)
                append(out/'predictions.jsonl',row);rows.append(row);mx.clear_cache()
            if index%8==7:print(json.dumps(dict(phase=phase,banks=index+1,resource=guard.check())),flush=True)
    aggregates={}
    for phase in checkpoints:
        subset=[r for r in rows if r['phase']==phase]
        aggregates[phase]=dict(object_token_exact=sum(r['object_token_exact'] for r in subset),total=len(subset),
            true_prefix_correct=sum(r['object_token_exact'] for r in subset if r['true_previous_subject']),true_prefix_total=32,
            mean_object_nll=sum(r['nll'] for r in subset)/len(subset))
    save(out/'summary.json',dict(completed=True,aggregates=aggregates,resource=guard.check()))


if __name__=='__main__':main()
