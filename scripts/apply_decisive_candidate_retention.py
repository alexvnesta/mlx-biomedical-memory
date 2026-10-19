"""Apply the pre-test candidate-retention amendment to an already running driver."""
import hashlib
import json
from pathlib import Path
import shutil
import time
from run_decisive_experiment import select


def main():
    for mode in ['text','memory']:
        training=Path(f'runs/decisive-{mode}-1201');receipt=training/'generated-selection-1024.json'
        while not receipt.exists():time.sleep(1)
        assert not list(Path('runs').glob('decisive-*-test/predictions.jsonl')),'Do not amend after held-out outputs'
        original=receipt.read_bytes();backup=training/'phase-only-selection-1024.json'
        if not backup.exists():backup.write_bytes(original)
        select(training,1024)
        (training/'candidate-retention-amendment.json').write_text(json.dumps(dict(
            applied_before_heldout=True,original_sha256=hashlib.sha256(backup.read_bytes()).hexdigest(),
            amended_sha256=hashlib.sha256(receipt.read_bytes()).hexdigest(),
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2)+'\n')
    gate=Path('runs/decisive-development-gate.json')
    while not gate.exists():time.sleep(1)
    original=json.loads(gate.read_text());selection={m:json.loads(Path(f'runs/decisive-{m}-1201/generated-selection.json').read_text())['selected'] for m in ['text','memory']}
    passed=selection['text']['reasoning_proofs']>=36 and selection['memory']['atomic_correct']>=22
    assert passed==original['passed'] and original['budget']==1024,'Retention changed the stopping decision'
    backup=Path('runs/decisive-development-gate-before-retention.json')
    if not backup.exists():backup.write_bytes(gate.read_bytes())
    gate.write_text(json.dumps(dict(original,selection=selection,candidate_retention_amended=True),indent=2)+'\n')


if __name__=='__main__':main()
