"""Exact graph traversal control for the stipulated rules; no language model."""
import hashlib
import json
from pathlib import Path
import time
from engram_poc.decisive_data import solution,atomic_solution,grade


def main():
    out=Path('runs/decisive-graph-reference');out.mkdir(exist_ok=False)
    data=Path('data/processed/decisive-v1/dataset.json');corpus=json.loads(data.read_text())
    cases=[c for s in ['iid','structure'] for c in corpus['splits'][s]]+corpus['interventions']
    rows=[]
    for case in cases:
        start=time.perf_counter_ns()
        for _ in range(100):answer=atomic_solution(case) if case['kind']=='atomic' else solution(case)
        seconds=(time.perf_counter_ns()-start)/1e9/100
        rows.append(dict(case_id=case['id'],kind=case['kind'],parsed=answer,seconds=seconds,**grade(case,answer)))
    assert len(rows)==352 and all(r['proof_valid'] for r in rows)
    (out/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    (out/'source.py').write_bytes(Path(__file__).read_bytes())
    (out/'decisive_data.py').write_bytes(Path('engram_poc/decisive_data.py').read_bytes())
    (out/'summary.json').write_text(json.dumps(dict(completed=True,rows=len(rows),
        dataset_sha256=hashlib.sha256(data.read_bytes()).hexdigest(),timing='100 traversals per world, mean wall time; Python-only, no LLM verbalization',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        semantics_sha256=hashlib.sha256(Path('engram_poc/decisive_data.py').read_bytes()).hexdigest()),indent=2)+'\n')


if __name__=='__main__':main()
