"""Evaluate trained memory candidates separately when initial fallback wins selection."""
import json
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    assert all(json.loads(Path(f'runs/decisive-{mode}-{seed}-test/summary.json').read_text())['completed'] for seed in [1201,1202] for mode in ['text','memory'])
    for seed in [1201,1202]:
        training=Path(f'runs/decisive-memory-{seed}')
        selection=json.loads((training/'generated-selection.json').read_text())
        candidates=[c for c in selection['candidates'] if c['step']>0]
        chosen=max(candidates,key=lambda x:(x['reasoning_proofs'],x['atomic_correct'],-x['weighted_dev_nll']))
        shutil.copyfile(training/chosen['checkpoint_name'],training/'trained-selected.safetensors')
        (training/'trained-only-selection.json').write_text(json.dumps(dict(candidates=candidates,selected=chosen,
            purpose='Pre-test registered trained-only diagnostic; initial fallback remains a separate primary result'),indent=2)+'\n')
        for variant in ['main','depth','training_audit']:
            suffix={'main':'-trained-test','depth':'-trained-depth-test','training_audit':'-training-audit'}[variant];out=Path(str(training)+suffix)
            if (out/'summary.json').exists():continue
            arguments=[sys.executable,'-u','scripts/evaluate_decisive_reader.py','--training',str(training),
                '--checkpoint-name','trained-selected.safetensors','--output',str(out)]
            if variant!='main':arguments+=['--supplemental-data','data/processed/decisive-v1/'+('depth-only.json' if variant=='depth' else 'training-audit.json')]
            if out.exists():arguments+=['--resume']
            print('RUN',*arguments,flush=True)
            with Path(str(out)+'.log').open('a') as log:subprocess.run(arguments,stdout=log,stderr=subprocess.STDOUT,check=True)


if __name__=='__main__':main()
