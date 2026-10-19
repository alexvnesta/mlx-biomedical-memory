"""Add joint adaptation, then evaluate all preselected composition conditions."""
import argparse
import json
from pathlib import Path

from run_composition_experiment import FEATURES,WARM,SEEDS,sha,run,select


ROOT=Path('runs/composition-joint-v1')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--development-only',action='store_true');args=p.parse_args()
    control_gate=Path('runs/composition-v1/development-gate.json')
    original=json.loads(control_gate.read_text());assert original['passed']
    budget=original['budget'];ROOT.mkdir(exist_ok=True)
    config=dict(controller_sha256=sha(__file__),control_gate_sha256=sha(control_gate),
        controller_dependency_sha256=sha('scripts/run_composition_experiment.py'),
        dataset_sha256=sha(FEATURES/'dataset.json'),feature_manifest_sha256=sha(FEATURES/'manifest.jsonl'),
        amendment_sha256=sha('docs/COMPOSITION-JOINT-AMENDMENT.md'),warm_start_sha256=sha(WARM),budget=budget,seeds=SEEDS)
    if (ROOT/'config.json').exists():assert json.loads((ROOT/'config.json').read_text())==config
    else:(ROOT/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    gate_path=ROOT/'development-gate.json'
    if not gate_path.exists():
        selections={}
        for seed in SEEDS:
            output=Path(f'runs/composition-joint-{seed}')
            complete=json.loads((output/'summary.json').read_text())['updates'] if (output/'summary.json').exists() else 0
            if complete<budget:
                arguments=['--mode','memory','--seed',seed,'--features',FEATURES,'--output',output,
                    '--updates',budget,'--maximum-updates',2048,'--warm-start',WARM,'--codec-lr',.00001,
                    '--protocol','docs/COMPOSITION-JOINT-AMENDMENT.md']
                if output.exists():arguments+=['--resume']
                run('train_decisive_reader.py',arguments,str(output)+'.log')
            selections[f'joint-{seed}']=select(output,budget)
        gate=dict(budget=budget,selection=selections,passed=all(c is not None for c in selections.values()))
        gate_path.write_text(json.dumps(gate,indent=2)+'\n')
    else:gate=json.loads(gate_path.read_text())
    assert gate['passed'],'Joint model failed readout retention; keep graph test closed'
    for key,item in gate['selection'].items():assert sha(Path('runs')/('composition-'+key)/'selected.safetensors')==item['checkpoint_sha256']
    if args.development_only:
        print('ALL DEVELOPMENT GATES PASSED; publish all three conditions before final generations',flush=True);return
    run('run_composition_experiment.py',[],ROOT/'control-tests.log')
    for seed in SEEDS:
        training=Path(f'runs/composition-joint-{seed}');output=Path(str(training)+'-test')
        if (output/'summary.json').exists():continue
        arguments=['--training',training,'--features',FEATURES,'--output',output,
            '--checkpoint-name','selected.safetensors','--include-support-readouts']
        if output.exists():arguments+=['--resume']
        run('evaluate_decisive_reader.py',arguments,str(output)+'.log')
    (ROOT/'summary.json').write_text(json.dumps(dict(completed=True,budget=budget,seeds=SEEDS),indent=2)+'\n')


if __name__=='__main__':main()
