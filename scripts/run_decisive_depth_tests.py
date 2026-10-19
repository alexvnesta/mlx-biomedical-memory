"""Evaluate the frozen depth-only control with the already selected readers."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from engram_poc.decisive_data import solution,grade


def main():
    dataset=Path('data/processed/decisive-v1/depth-only.json');corpus=json.loads(dataset.read_text())
    assert all(json.loads(Path(f'runs/decisive-{mode}-{seed}-test/summary.json').read_text())['completed'] for seed in [1201,1202] for mode in ['text','memory'])
    for seed in [1201,1202]:
        for mode in ['text','memory']:
            training=Path(f'runs/decisive-{mode}-{seed}')
            assert json.loads(Path(str(training)+'-test/summary.json').read_text())['completed'],'Finish the registered serial run first'
            output=Path(str(training)+'-depth-test')
            if (output/'summary.json').exists():continue
            args=[sys.executable,'-u','scripts/evaluate_decisive_reader.py','--training',str(training),
                '--checkpoint-name','selected.safetensors','--supplemental-data',str(dataset),'--output',str(output)]
            if output.exists():args+=['--resume']
            print('RUN',*args,flush=True)
            with Path(str(output)+'.log').open('a') as log:subprocess.run(args,stdout=log,stderr=subprocess.STDOUT,check=True)
    graph=Path('runs/decisive-graph-reference');rows=[]
    for case in corpus['cases']:
        start=time.perf_counter_ns()
        for _ in range(100):answer=solution(case)
        seconds=(time.perf_counter_ns()-start)/1e9/100
        rows.append(dict(case_id=case['id'],kind=case['kind'],parsed=answer,seconds=seconds,**grade(case,answer)))
    assert len(rows)==96 and all(r['proof_valid'] for r in rows)
    (graph/'depth-predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    (graph/'depth-source.py').write_bytes(Path(__file__).read_bytes())
    (graph/'depth-summary.json').write_text(json.dumps(dict(completed=True,rows=96,
        dataset_sha256=hashlib.sha256(dataset.read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2)+'\n')


if __name__=='__main__':main()
