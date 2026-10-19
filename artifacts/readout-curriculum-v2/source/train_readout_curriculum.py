"""Fresh-bank LongCat readout curriculum. No reasoning labels are used."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import re
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map
from mlx_lm import stream_generate
from mlx_lm.sample_utils import make_sampler

from engram_poc.decisive_model import load_reader
from engram_poc.decisive_data import fields, atomic_solution
from engram_poc.resource_guard import ResourceGuard
from engram_poc.longcat_runtime import append, save, snapshot_source
from evaluate_decisive_reader import parse


def pair_split(subject, object):
    value=(5*subject+3*object)%8
    return 'dev' if value==0 else 'test' if value==1 else 'train'


def make_case(seed, count, split='train'):
    rng=random.Random(seed);ids=rng.sample(range(10),count)
    records=[dict(id=f'F{i}',subject=rng.randrange(16),object=rng.randrange(16),
        sign=rng.choice([-1,1]),species=rng.randrange(2),variant=rng.randrange(2),source=rng.randrange(4)) for i in ids]
    for r in records:
        while pair_split(r['subject'],r['object'])!=split:
            r['subject']=rng.randrange(16);r['object']=rng.randrange(16)
    return dict(id=f'fresh-{seed}-{count}',kind='atomic',records=records,record_id=rng.choice(records)['id'])


def encode(tok, case):
    text=('Read the requested latent record. Each @ stores one complete record, with subject, object, sign, '
        'species, variant, source. Return JSON with keys record, subject, object, sign, species, variant, source.\n')
    for r in case['records']:text+=r['id']+': @\n'
    text+='READ '+case['record_id']+'. Return only its JSON.'
    prompt=tok.apply_chat_template([dict(role='user',content=text)],tokenize=False,add_generation_prompt=True,enable_thinking=False)
    ids=tok.encode(prompt,add_special_tokens=False);positions=[];cursor=0
    for r in case['records']:
        end=prompt.index(r['id']+': @',cursor)+len(r['id']+': @');cursor=end
        positions.append(len(tok.encode(prompt[:end],add_special_tokens=False))-1)
    target=json.dumps(atomic_solution(case),separators=(',',':'))
    return dict(prompt=prompt,ids=ids,positions=positions,target=tok.encode(target,add_special_tokens=False)+[2])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('runs/readout-curriculum-v2'))
    p.add_argument('--updates',type=int,default=2048)
    p.add_argument('--seed',type=int,default=2301)
    p.add_argument('--eval-every',type=int,default=128)
    p.add_argument('--eval-count',type=int,default=32)
    p.add_argument('--start-count',type=int,default=1)
    p.add_argument('--checkpoint',type=Path)
    args=p.parse_args();out=args.output;out.mkdir(exist_ok=False)
    save(out/'config.json',dict(seed=args.seed,updates=args.updates,model='LongCat-Flash-Lite-4bit',
        source_sha256=snapshot_source(__file__,out/'source'),training='2 fresh independent banks per update; atomic JSON only',
        stages=[1,4,10],advancement='at least 31/32 exact generated development records',
        learning_rates=dict(lora=.00005,codec=.001),selection='generated exact record accuracy',
        development_seeds='41000000 + record_count * 10000 + index',test_seeds='not yet evaluated',
        pair_holdout='(5*subject+3*object)%8: dev=0, test=1, train=2..7, across every bank record',
        checkpoint=str(args.checkpoint) if args.checkpoint else None))
    guard=ResourceGuard();mx.random.seed(args.seed)
    print('Loading full frozen prefix and trainable tail',flush=True)
    model,tok=load_reader('memory');mx.eval(model.parameters())
    if args.checkpoint:model.load_weights(str(args.checkpoint),strict=False)
    print('Loaded '+json.dumps(guard.check()),flush=True)
    opts=[optim.AdamW(.001,weight_decay=0),optim.AdamW(.00005,weight_decay=0)]
    optimizer=optim.MultiOptimizer(opts,[lambda k,v:k.startswith('memory.')])
    def loss_fn(network,h,f,pos,start,target,weights):
        logits=network.tail(h,f,pos,loss_start=start).astype(mx.float32)
        # Content and syntax both have to be predicted, with no prior gold field access beyond causal teacher forcing.
        ce=nn.losses.cross_entropy(logits,mx.array([target]),reduction='none')[0]
        w=mx.array(weights);return mx.sum(ce*w)/mx.sum(w)
    vg=nn.value_and_grad(model,loss_fn)
    def generate(case):
        e=encode(tok,case);model.begin(mx.array(fields(case['records'])),e['positions'])
        pieces=[]
        for response in stream_generate(model,tok,e['ids'],sampler=make_sampler(temp=0),max_tokens=100):
            pieces.append(response.text)
            if parse(''.join(pieces),'atomic') is not None:break
        return ''.join(pieces),parse(''.join(pieces),'atomic')
    def validate(step,count):
        full=dict(tree_flatten(model.trainable_parameters()))
        reduced={k:v.astype(mx.bfloat16) for k,v in full.items()}
        model.load_weights(list(reduced.items()),strict=False);mx.eval(model.trainable_parameters())
        correct=0;field_correct={k:0 for k in atomic_solution(make_case(0,1))}
        for i in range(args.eval_count):
            case=make_case(41000000+count*10000+i,count,'dev');text,parsed=generate(case);expected=atomic_solution(case)
            correct+=parsed==expected
            for k,v in expected.items():field_correct[k]+=(parsed or {}).get(k)==v
            append(out/'development.jsonl',dict(step=step,count=count,case=case,generation=text,parsed=parsed,
                expected=expected,correct=parsed==expected))
            mx.clear_cache()
        mx.save_safetensors(str(out/f'step-{step:04d}.safetensors'),reduced)
        model.load_weights(list(full.items()),strict=False);mx.eval(model.trainable_parameters())
        result=dict(step=step,count=count,correct=correct,total=args.eval_count,field_correct=field_correct,resource=guard.check())
        append(out/'validation.jsonl',result);print('VALIDATION '+json.dumps(result),flush=True)
        return correct
    count=args.start_count;best=-1;start=time.perf_counter();completed=0
    try:
        for step in range(1,args.updates+1):
            accumulated=None;loss=0
            for draw in range(2):
                case=make_case(args.seed*1000000+step*2+draw,count);e=encode(tok,case)
                target_text=json.dumps(atomic_solution(case),separators=(',',':'))
                encoded=tok._tokenizer(target_text,add_special_tokens=False,return_offsets_mapping=True)
                spans=[m.span(1) for m in re.finditer(r'"(N\d+|S\d+|UP|DOWN|human|mouse|lof|gof)"',target_text)]
                weights=[2. if any(a<end and b>start for start,end in spans) else .1 for a,b in encoded['offset_mapping']]+[.1]
                model.begin(None,[]) # Prefix never depends on values or requested labels.
                h=model.prefix(mx.array([e['ids']+e['target'][:-1]]));mx.eval(h)
                value,grad=vg(model,h,mx.array(fields(case['records'])),e['positions'],len(e['ids'])-1,e['target'],weights)
                mx.eval(value,grad);loss+=value.item()/2
                accumulated=grad if accumulated is None else tree_map(lambda a,b:a+b,accumulated,grad)
                del h,value,grad;mx.clear_cache()
            clipped,norm=optim.clip_grad_norm(tree_map(lambda x:x/2,accumulated),max_norm=1)
            assert __import__('math').isfinite(norm.item())
            for opt,lr in zip(opts,[.001,.00005]):opt.learning_rate=lr*min(1,step/16)
            optimizer.update(model,clipped);mx.eval(model.trainable_parameters(),optimizer.state);completed=step
            row=dict(step=step,count=count,loss=loss,gradient_norm=norm.item(),seconds=time.perf_counter()-start)
            if step%16==0 or step==1:row['resource']=guard.check();print(json.dumps(row),flush=True)
            append(out/'training.jsonl',row)
            if step%args.eval_every==0 or step==args.updates:
                score=validate(step,count)
                if score>best:
                    best=score;(out/'selected.safetensors').write_bytes((out/f'step-{step:04d}.safetensors').read_bytes())
                    save(out/'selection.json',dict(step=step,count=count,correct=score,total=args.eval_count))
                if score/args.eval_count>=.96875 and count<10:
                    count=4 if count==1 else 10;best=-1;print('ADVANCE '+str(count),flush=True)
                elif score/args.eval_count>=.96875 and count==10:
                    print('DEVELOPMENT GATE PASSED; freeze checkpoint before fresh causal test',flush=True);break
    finally:
        mx.save_safetensors(str(out/'last.safetensors'),dict(tree_flatten(model.trainable_parameters())))
        save(out/'summary.json',dict(completed=completed==args.updates or (count==10 and best/args.eval_count>=.96875),
            updates=completed,count=count,seconds=time.perf_counter()-start,resource=guard.check()))


if __name__=='__main__':main()
