"""Train matched LongCat readers using frozen causal prefix features."""

import argparse
from datetime import datetime,timezone
import hashlib
import json
import math
from pathlib import Path
import random
import re
import time
import numpy as np

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten,tree_unflatten,tree_map

from engram_poc.decisive_model import load_reader
from engram_poc.decisive_prompts import target_text
from engram_poc.longcat_runtime import snapshot_source,save,append
from engram_poc.resource_guard import ResourceGuard
from cache_decisive_features import digest_frozen


def target_weights(tokenizer,text):
    encoded=tokenizer._tokenizer(text,add_special_tokens=False,return_offsets_mapping=True)
    spans=[m.span(1) for m in re.finditer(r'"(F\d+|N\d+|S\d+|UP|DOWN|BOTH|UNKNOWN|human|mouse|lof|gof)"',text)]
    answer=re.search(r'"answer":"([A-Z]+)"',text)
    weights=[]
    for start,end in encoded['offset_mapping']:
        w=2. if any(start<b and end>a for a,b in spans) else .25
        if answer and start<answer.end(1) and end>answer.start(1):w=4.
        weights.append(w)
    return weights+[1.]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features',type=Path,default=Path('runs/decisive-features-v1'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--mode',choices=['text','memory'],required=True)
    parser.add_argument('--seed',type=int,default=1201)
    parser.add_argument('--updates',type=int,default=512)
    parser.add_argument('--lora-lr',type=float,default=.00003)
    parser.add_argument('--codec-lr',type=float,default=.001)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--warm-start',type=Path)
    parser.add_argument('--freeze-codec',action='store_true')
    parser.add_argument('--protocol',type=Path,default=Path('docs/DECISIVE-PROTOCOL.md'))
    parser.add_argument('--maximum-updates',type=int,default=1024)
    args=parser.parse_args();out=args.output
    assert json.loads((args.features/'summary.json').read_text())['completed']
    corpus=json.loads((args.features/'dataset.json').read_text())
    feature_config=json.loads((args.features/'config.json').read_text())
    if out.exists() and not args.resume:raise ValueError('Preserve completed training outputs')
    out.mkdir(exist_ok=True)
    if not args.resume:
        config=dict(mode=args.mode,seed=args.seed,started_utc=datetime.now(timezone.utc).isoformat(),
            source_sha256=snapshot_source(__file__,out/'source'),dataset_sha256=feature_config['dataset_sha256'],
            feature_manifest_sha256=hashlib.sha256((args.features/'manifest.jsonl').read_bytes()).hexdigest(),
            maximum_updates=args.maximum_updates,lora_learning_rate=args.lora_lr,codec_learning_rate=args.codec_lr,
            gradient_accumulation=2,atomic_warmup_updates=64,
            later_mixture='one reasoning and one atomic example per optimizer update',
            checkpoint_selection='NLL candidate plus initial and final candidates; generated development proof selects the final published reader',
            published_precision='BF16 adapters and codec; frozen native 4-bit backbone',
            protocol_sha256=hashlib.sha256(args.protocol.read_bytes()).hexdigest(),
            warm_start=str(args.warm_start) if args.warm_start else None,
            warm_start_sha256=hashlib.sha256(args.warm_start.read_bytes()).hexdigest() if args.warm_start else None,
            frozen_codec=args.freeze_codec,
            dependency_sha256=hashlib.sha256(Path('scripts/cache_decisive_features.py').read_bytes()).hexdigest())
        save(out/'config.json',config)
        (out/'protocol.md').write_bytes(args.protocol.read_bytes())
        (out/'cache_decisive_features.py').write_bytes(Path('scripts/cache_decisive_features.py').read_bytes())
    else:config=json.loads((out/'config.json').read_text())
    assert config['mode']==args.mode and config['seed']==args.seed and args.updates<=config['maximum_updates']
    assert args.lora_lr==config['lora_learning_rate'] and args.codec_lr==config['codec_learning_rate']
    assert args.freeze_codec==config.get('frozen_codec',False)
    if args.warm_start:assert config['warm_start_sha256']==hashlib.sha256(args.warm_start.read_bytes()).hexdigest()
    assert config['dataset_sha256']==feature_config['dataset_sha256']
    assert config['feature_manifest_sha256']==hashlib.sha256((args.features/'manifest.jsonl').read_bytes()).hexdigest()
    if args.resume:
        source_hash=hashlib.sha256()
        for path in [Path(__file__),*sorted(Path('engram_poc').glob('*.py'))]:
            source_hash.update(path.name.encode());source_hash.update(path.read_bytes())
        assert source_hash.hexdigest()==config['source_sha256'],'Resume source changed'
    guard=ResourceGuard();mx.random.seed(args.seed)
    print('Loading frozen tail and '+args.mode+' reader',flush=True);model,tokenizer=load_reader(args.mode,tail_only=True)
    if args.warm_start:
        warm=mx.load(str(args.warm_start))
        assert set(warm)==set(dict(tree_flatten(model.trainable_parameters())))
        model.load_weights([(k,v if args.freeze_codec and k.startswith('memory.') else v.astype(mx.float32)) for k,v in warm.items()],strict=False)
    if args.freeze_codec:
        assert args.mode=='memory' and args.warm_start,'A frozen codec requires a proved memory warm start'
        model.memory.freeze()
    def codec_digest():
        if not args.freeze_codec:return None
        h=hashlib.sha256()
        for name,array in tree_flatten(model.memory.parameters()):
            h.update(name.encode());h.update(str(array.shape).encode());h.update(str(array.dtype).encode())
            h.update(memoryview(np.array(array.reshape(-1).view(mx.uint8))))
        return h.hexdigest()
    codec_before=codec_digest()
    def checkpoint_parameters():
        values=dict(tree_flatten(model.trainable_parameters()))
        if args.freeze_codec:values.update({'memory.'+k:v for k,v in tree_flatten(model.memory.parameters())})
        return values
    trainable=dict(tree_flatten(model.trainable_parameters()))
    assert trainable and all(name.startswith('memory.') or '.lora_a' in name or '.lora_b' in name for name in trainable)
    count=sum(x.size for x in trainable.values());mx.eval(model.parameters())
    save(out/'capacity.json',dict(trainable_parameters=count,shapes={k:list(v.shape) for k,v in trainable.items()},
        frozen_codec_parameters=sum(v.size for _,v in tree_flatten(model.memory.parameters())) if args.freeze_codec else 0))
    before=digest_frozen(model)
    optimizers=[optim.AdamW(args.lora_lr,weight_decay=0.)];learning_rates=[args.lora_lr]
    has_codec_optimizer=args.mode=='memory' and not args.freeze_codec
    if has_codec_optimizer:
        optimizers.insert(0,optim.AdamW(args.codec_lr,weight_decay=0.));learning_rates.insert(0,args.codec_lr)
    optimizer=optim.MultiOptimizer(optimizers,[lambda name,value:name.startswith('memory.')] if has_codec_optimizer else [])
    manifest=[json.loads(line) for line in (args.features/'manifest.jsonl').read_text().splitlines()]
    selected={r['id']:r for r in manifest if r['mode']==args.mode}
    examples={c['id']:c for split in ['train','dev'] for c in corpus['splits'][split]}
    for identifier,row in selected.items():
        expected=target_text(examples[identifier]);weights=target_weights(tokenizer,expected)
        assert len(weights)==len(row['target']);row['weights']=weights
    pools={kind:[c['id'] for c in corpus['splits']['train'] if c['kind']==kind] for kind in ['reason','atomic']}
    rng=random.Random(args.seed+8800);schedule=[]
    for step in range(1,args.updates+1):
        schedule.append([rng.choice(pools['atomic']),rng.choice(pools['atomic'] if step<=64 else pools['reason'])])
    best=math.inf;begin_step=0
    if args.resume:
        model.load_weights(str(out/'last.safetensors'),strict=False)
        optimizer.state=tree_unflatten(list(mx.load(str(out/'optimizer.safetensors')).items()))
        state=json.loads((out/'state.json').read_text());begin_step=state['step'];best=state['best_dev_nll']
        assert state['frozen_digest']==before
        assert codec_digest()==codec_before
    started=time.perf_counter()

    def batch(identifier):
        row=selected[identifier];path=Path(row['path'])
        # Cached features are local derived model states; their digests are public.
        assert hashlib.sha256(path.read_bytes()).hexdigest()==row['feature_sha256']
        return row,mx.load(str(path))['hidden']

    def loss_fn(network,h,f,positions,start,target,weights):
        logits=network.tail(h,f,positions,loss_start=start).astype(mx.float32)
        ce=nn.losses.cross_entropy(logits,mx.array([target]),reduction='none')[0]
        w=mx.array(weights);return mx.sum(ce*w)/mx.sum(w)

    vg=nn.value_and_grad(model,loss_fn)

    def validation(step):
        losses=[]
        for case in corpus['splits']['dev']:
            row,h=batch(case['id']);value=loss_fn(model,h,mx.array(row['fields']),row['positions'],row['loss_start'],row['target'],row['weights'])
            mx.eval(value);losses.append(value.item());del h;mx.clear_cache()
            append(out/'development-loss.jsonl',dict(step=step,case_id=case['id'],kind=case['kind'],weighted_nll=losses[-1]))
        return sum(losses)/len(losses)

    def checkpoint(step):
        mx.save_safetensors(str(out/'last.safetensors'),checkpoint_parameters())
        mx.save_safetensors(str(out/'optimizer.safetensors'),dict(tree_flatten(optimizer.state)))
        save(out/'state.json',dict(step=step,best_dev_nll=best,frozen_digest=before))

    def select(step,dev):
        nonlocal best
        if dev<best:
            best=dev
            mx.save_safetensors(str(out/'best.safetensors'),
                {k:v.astype(mx.bfloat16) for k,v in checkpoint_parameters().items()})
            save(out/'selection.json',dict(step=step,weighted_dev_nll=dev,
                checkpoint_sha256=hashlib.sha256((out/'best.safetensors').read_bytes()).hexdigest()))

    if not args.resume:
        initial=validation(0);select(0,initial)
        mx.save_safetensors(str(out/'initial.safetensors'),
            {k:v.astype(mx.bfloat16) for k,v in checkpoint_parameters().items()})
        print('Initial development '+str(initial),flush=True)

    completed=begin_step
    try:
        for step in range(begin_step+1,args.updates+1):
            total=0.;accumulated=None
            for identifier in schedule[step-1]:
                row,h=batch(identifier)
                value,grad=vg(model,h,mx.array(row['fields']),row['positions'],row['loss_start'],row['target'],row['weights'])
                mx.eval(value,grad);total+=value.item()/2
                if accumulated is None:accumulated=grad
                else:accumulated=tree_map(lambda a,b:a+b,accumulated,grad)
                del h,value,grad;mx.clear_cache()
            accumulated=tree_map(lambda x:x/2,accumulated)
            clipped,norm=optim.clip_grad_norm(accumulated,max_norm=1.)
            assert math.isfinite(norm.item()),'Nonfinite reader gradient'
            for opt,rate in zip(optimizers,learning_rates):opt.learning_rate=rate*min(1.,step/16)
            optimizer.update(model,clipped);mx.eval(model.trainable_parameters(),optimizer.state)
            completed=step
            row=dict(step=step,weighted_loss=total,gradient_norm=norm.item(),elapsed_seconds=time.perf_counter()-started)
            if step%16==0 or step==1:row.update(resource=guard.check());print(json.dumps(row),flush=True)
            append(out/'training.jsonl',row)
            if step%64==0 or step==args.updates:
                dev=validation(step);select(step,dev)
                print('Development '+json.dumps(dict(step=step,weighted_nll=dev,best=best)),flush=True)
                checkpoint(step)
    except BaseException:
        checkpoint(completed);raise
    after=digest_frozen(model);assert before==after
    codec_after=codec_digest();assert codec_before==codec_after
    final={k:v.astype(mx.bfloat16) for k,v in checkpoint_parameters().items()}
    mx.save_safetensors(str(out/f'final-{completed:04d}.safetensors'),final)
    mx.save_safetensors(str(out/'final.safetensors'),final)
    (out/f'best-{completed:04d}.safetensors').write_bytes((out/'best.safetensors').read_bytes())
    summary=dict(completed=True,updates=completed,seconds=time.perf_counter()-started,
        trainable_parameters=count,best_dev_nll=best,resource=guard.check(),
        frozen_codec=dict(before=codec_before,after=codec_after,unchanged=codec_before==codec_after),
        frozen_constants=dict(before=before,after=after,unchanged=True))
    save(out/f'summary-{completed:04d}.json',summary);save(out/'summary.json',summary)
    save(out/f'nll-selection-{completed:04d}.json',json.loads((out/'selection.json').read_text()))
    print('COMPLETED '+json.dumps(summary),flush=True)


if __name__=='__main__':main()
