"""Run the registered serial training, development selection and held-out study."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys


def run(script,arguments,log):
    print('RUN',script,*map(str,arguments),flush=True)
    with Path(log).open('a') as stream:
        subprocess.run([sys.executable,'-u','scripts/'+script,*map(str,arguments)],stdout=stream,stderr=subprocess.STDOUT,check=True)


def select(training,budget):
    by_hash={};candidates=[]
    for name in ['initial.safetensors',f'best-{budget:04d}.safetensors',f'final-{budget:04d}.safetensors']:
        digest=hashlib.sha256((training/name).read_bytes()).hexdigest()
        output=Path(str(training)+f'-dev-{budget}-{name.split(".")[0]}')
        if digest in by_hash:
            candidates.append(dict(by_hash[digest],checkpoint_name=name));continue
        if not (output/'summary.json').exists():
            args=['--training',training,'--development','--checkpoint-name',name,'--output',output]
            if output.exists():args+=['--resume']
            run('evaluate_decisive_reader.py',args,str(output)+'.log')
        rows=[json.loads(x) for x in (output/'predictions.jsonl').read_text().splitlines()]
        assert len(rows)==72
        reason=[r for r in rows if r['kind']=='reason'];atomic=[r for r in rows if r['kind']=='atomic']
        item=dict(checkpoint_name=name,checkpoint_sha256=digest,evaluation=str(output),
            reasoning_proofs=sum(r['proof_valid'] for r in reason),reasoning_correct=sum(r['correct'] for r in reason),
            atomic_correct=sum(r['correct'] for r in atomic),reasoning_total=len(reason),atomic_total=len(atomic))
        # Selection's final tie-break is the checkpoint's actual development loss.
        losses=[json.loads(x) for x in (training/'development-loss.jsonl').read_text().splitlines()]
        if name.startswith('initial'):step=0
        elif name.startswith('final'):step=budget
        else:
            grouped={s:sum(x['weighted_nll'] for x in losses if x['step']==s)/72 for s in {x['step'] for x in losses}}
            step=min(grouped,key=grouped.get)
        item['step']=step;item['weighted_dev_nll']=sum(x['weighted_nll'] for x in losses if x['step']==step)/72
        by_hash[digest]=item;candidates.append(item)
    # Retain every already scored candidate from an earlier budget. Extending
    # training must not discard a better generated-development checkpoint.
    for previous in sorted(training.glob('generated-selection-[0-9]*.json')):
        prior=json.loads(previous.read_text())
        for candidate in prior['candidates']:
            if candidate['checkpoint_sha256'] not in {c['checkpoint_sha256'] for c in candidates}:
                candidates.append(candidate)
    chosen=max(candidates,key=lambda x:(x['reasoning_proofs'],x['atomic_correct'],-x['weighted_dev_nll']))
    shutil.copyfile(training/chosen['checkpoint_name'],training/'selected.safetensors')
    receipt=dict(budget=budget,candidates=candidates,selected=chosen)
    (training/f'generated-selection-{budget}.json').write_text(json.dumps(receipt,indent=2)+'\n')
    (training/'generated-selection.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print('SELECTED',training,json.dumps(chosen),flush=True)
    return chosen


def train(mode,seed,budget):
    output=Path(f'runs/decisive-{mode}-{seed}')
    complete=json.loads((output/'summary.json').read_text())['updates'] if (output/'summary.json').exists() else 0
    if complete<budget:
        args=['--mode',mode,'--seed',seed,'--updates',budget,'--output',output]
        if output.exists():args+=['--resume']
        run('train_decisive_reader.py',args,str(output)+'.log')
    return select(output,budget)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--development-only',action='store_true');args=parser.parse_args()
    gate=Path('runs/decisive-development-gate.json')
    if gate.exists():budget=json.loads(gate.read_text())['budget']
    else:
        selected={m:train(m,1201,512) for m in ['text','memory']};budget=512
        if selected['text']['reasoning_proofs']<36 or selected['memory']['atomic_correct']<22:
            budget=1024;selected={m:train(m,1201,budget) for m in ['text','memory']}
        gate.write_text(json.dumps(dict(budget=budget,selection=selected,
            passed=selected['text']['reasoning_proofs']>=36 and selected['memory']['atomic_correct']>=22),indent=2)+'\n')
    for mode in ['text','memory']:train(mode,1202,budget)
    if args.development_only:return
    for seed in [1201,1202]:
        for mode in ['text','memory']:
            training=Path(f'runs/decisive-{mode}-{seed}');out=Path(str(training)+'-test')
            if (out/'summary.json').exists():continue
            arguments=['--training',training,'--checkpoint-name','selected.safetensors','--output',out]
            if out.exists():arguments+=['--resume']
            run('evaluate_decisive_reader.py',arguments,str(out)+'.log')


if __name__=='__main__':main()
