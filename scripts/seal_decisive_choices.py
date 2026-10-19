"""Record exact candidate choices before any held-out neural generation exists."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    paths=[Path(f'runs/decisive-{mode}-{seed}/generated-selection.json') for mode in ['text','memory'] for seed in [1201,1202]]
    while not all(p.exists() and json.loads(p.read_text())['budget']==1024 for p in paths):time.sleep(.2)
    assert not list(Path('runs').glob('decisive-*-test/predictions.jsonl')),'A held-out neural output already exists'
    choices={}
    for path in paths:
        run=path.parent;selection=json.loads(path.read_text());chosen=selection['selected']
        assert digest(run/'selected.safetensors')==chosen['checkpoint_sha256']
        choices[run.name]=dict(selected=chosen,selection_sha256=digest(path))
        if '-memory-' in run.name:
            candidates=[c for c in selection['candidates'] if c['step']>0]
            fitted=max(candidates,key=lambda c:(c['reasoning_proofs'],c['atomic_correct'],-c['weighted_dev_nll']))
            assert digest(run/fitted['checkpoint_name'])==fitted['checkpoint_sha256']
            choices[run.name]['trained_diagnostic']=fitted
    inputs={name:digest('data/processed/decisive-v1/'+name) for name in ['dataset.json','depth-only.json','training-audit.json']}
    sources={str(p):digest(p) for p in [Path('scripts/train_decisive_reader.py'),Path('scripts/evaluate_decisive_reader.py'),*sorted(Path('engram_poc').glob('*.py'))]}
    seal=dict(sealed_utc=datetime.now(timezone.utc).isoformat(),before_any_heldout_neural_prediction=True,
        git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),choices=choices,inputs=inputs,sources=sources,
        source_sha256=digest(__file__))
    Path('runs/decisive-pretest-seal.json').write_text(json.dumps(seal,indent=2)+'\n')
    Path('runs/decisive-pretest-seal-source.py').write_bytes(Path(__file__).read_bytes())
    print('Pre-test checkpoint and input seal written',flush=True)


if __name__=='__main__':main()
