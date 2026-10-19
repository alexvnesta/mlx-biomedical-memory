"""Serial, readout-gated comparison of new graph compositions."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys


FEATURES=Path('runs/composition-features-v1')
ROOT=Path('runs/composition-v1')
WARM=Path('runs/readout-curriculum-v5/selected.safetensors')
SEEDS=[2401,2402]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(script,args,log):
    print('RUN',script,*map(str,args),flush=True)
    with Path(log).open('a') as stream:
        subprocess.run([sys.executable,'-u','scripts/'+script,*map(str,args)],stdout=stream,stderr=subprocess.STDOUT,check=True)


def choose(candidates):
    eligible=[c for c in candidates if c['step']>0 and c['atomic_correct']>=63 and c['atomic_total']==64]
    return max(eligible,key=lambda c:(c['reasoning_proofs'],c['atomic_correct'],-c['weighted_dev_nll'])) if eligible else None


def select(training,budget):
    candidates=[];by_hash={}
    for previous in sorted(training.glob('generated-selection-*.json')):
        for candidate in json.loads(previous.read_text())['candidates']:
            by_hash[candidate['checkpoint_sha256']]=candidate
    nll=json.loads((training/f'nll-selection-{budget:04d}.json').read_text())
    losses=[json.loads(x) for x in (training/'development-loss.jsonl').read_text().splitlines()]
    for name,step in [(f'best-{budget:04d}.safetensors',nll['step']),(f'final-{budget:04d}.safetensors',budget)]:
        digest=sha(training/name)
        if step==0:continue
        if digest in by_hash:continue
        output=Path(str(training)+f'-dev-{budget}-{name.removesuffix(".safetensors")}')
        if not (output/'summary.json').exists():
            args=['--training',training,'--features',FEATURES,'--output',output,'--development','--checkpoint-name',name]
            if output.exists():args+=['--resume']
            run('evaluate_decisive_reader.py',args,str(output)+'.log')
        rows=[json.loads(x) for x in (output/'predictions.jsonl').read_text().splitlines()]
        atomic=[r for r in rows if r['kind']=='atomic'];reason=[r for r in rows if r['kind']=='reason']
        assert len(rows)==112 and len(atomic)==64 and len(reason)==48
        selected_losses=[r['weighted_nll'] for r in losses if r['step']==step];assert len(selected_losses)==112
        by_hash[digest]=dict(checkpoint_name=name,checkpoint_sha256=digest,evaluation=str(output),step=step,
            reasoning_proofs=sum(r['proof_valid'] for r in reason),reasoning_correct=sum(r['correct'] for r in reason),
            reasoning_total=48,atomic_correct=sum(r['correct'] for r in atomic),atomic_total=64,
            weighted_dev_nll=sum(selected_losses)/112)
    candidates=list(by_hash.values());chosen=choose(candidates)
    if chosen:shutil.copyfile(training/chosen['checkpoint_name'],training/'selected.safetensors')
    receipt=dict(budget=budget,candidates=candidates,selected=chosen)
    (training/f'generated-selection-{budget}.json').write_text(json.dumps(receipt,indent=2)+'\n')
    (training/'generated-selection.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print('SELECTION',training,json.dumps(chosen),flush=True);return chosen


def train(mode,seed,budget):
    output=Path(f'runs/composition-{mode}-{seed}')
    complete=json.loads((output/'summary.json').read_text())['updates'] if (output/'summary.json').exists() else 0
    if complete<budget:
        args=['--mode',mode,'--seed',seed,'--features',FEATURES,'--output',output,'--updates',budget,
            '--maximum-updates',2048,'--protocol','docs/COMPOSITION-PROTOCOL.md']
        if mode=='memory':args+=['--warm-start',WARM,'--freeze-codec']
        if output.exists():args+=['--resume']
        run('train_decisive_reader.py',args,str(output)+'.log')
    return select(output,budget)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--development-only',action='store_true');args=p.parse_args()
    assert json.loads((FEATURES/'summary.json').read_text())['completed']
    assert sha(WARM)=='3d913d797cf7209590c88336a014ca44d6a6197fe422f2ecc92d01ef9f3bb3c1'
    ROOT.mkdir(exist_ok=True)
    config=dict(controller_sha256=sha(__file__),dataset_sha256=sha(FEATURES/'dataset.json'),
        feature_manifest_sha256=sha(FEATURES/'manifest.jsonl'),protocol_sha256=sha('docs/COMPOSITION-PROTOCOL.md'),
        warm_start_sha256=sha(WARM),seeds=SEEDS,maximum_updates=2048)
    if (ROOT/'config.json').exists():assert json.loads((ROOT/'config.json').read_text())==config
    else:(ROOT/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    gate_path=ROOT/'development-gate.json'
    if not gate_path.exists():
        budget=1024;selections={}
        for mode in ['memory','text']:selections[f'{mode}-2401']=train(mode,2401,budget)
        if (selections['text-2401'] is None or selections['text-2401']['reasoning_proofs']<24 or selections['memory-2401'] is None):
            budget=2048
            for mode in ['memory','text']:selections[f'{mode}-2401']=train(mode,2401,budget)
        for mode in ['memory','text']:selections[f'{mode}-2402']=train(mode,2402,budget)
        if budget==1024 and any(c is None for c in selections.values()):
            budget=2048
            for seed in SEEDS:
                for mode in ['memory','text']:selections[f'{mode}-{seed}']=train(mode,seed,budget)
        gate=dict(budget=budget,selection=selections,passed=all(c is not None for c in selections.values()))
        gate_path.write_text(json.dumps(gate,indent=2)+'\n')
    else:gate=json.loads(gate_path.read_text())
    assert gate['passed'],'Do not judge reasoning: a development readout-retention gate failed'
    for key,item in gate['selection'].items():assert sha(Path('runs')/('composition-'+key)/'selected.safetensors')==item['checkpoint_sha256']
    if args.development_only:
        print('DEVELOPMENT COMPLETE; freeze/publish selected checkpoint and corpus receipts before final evaluation',flush=True);return
    for seed in SEEDS:
        for mode in ['text','memory']:
            training=Path(f'runs/composition-{mode}-{seed}');output=Path(str(training)+'-test')
            if (output/'summary.json').exists():continue
            arguments=['--training',training,'--features',FEATURES,'--output',output,
                '--checkpoint-name','selected.safetensors','--include-support-readouts']
            if output.exists():arguments+=['--resume']
            run('evaluate_decisive_reader.py',arguments,str(output)+'.log')
    (ROOT/'summary.json').write_text(json.dumps(dict(completed=True,budget=gate['budget'],seeds=SEEDS),indent=2)+'\n')


if __name__=='__main__':main()
