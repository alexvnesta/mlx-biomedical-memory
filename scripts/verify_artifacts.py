"""Portable verification of published checksums and comparison provenance."""

import hashlib
import json
from pathlib import Path


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root=Path('artifacts')
    registered=json.loads((root/'SHA256SUMS.json').read_text())
    actual={str(p) for p in root.rglob('*') if p.is_file() and p.name!='SHA256SUMS.json'}
    assert set(registered)==actual,'Artifact inventory differs from checksum registry'
    for path,expected in registered.items():assert digest(Path(path))==expected,path
    for directory,script in [('comparison-v1','run_comparison.py'),
            ('comparison-latency','benchmark_comparison_latency.py'),
            ('comparison-diagnostics','diagnose_comparison_memory.py'),
            ('comparison-abstention','audit_abstention.py')]:
        run=root/directory
        if not run.exists():continue
        config=json.loads((run/'config.json').read_text())
        sources=[run/'source'/script,*sorted(p for p in (run/'source').glob('*.py') if p.name!=script)]
        code_hash=hashlib.sha256()
        for path in sources:code_hash.update(path.name.encode());code_hash.update(path.read_bytes())
        assert code_hash.hexdigest()==config['source_sha256'],directory+' source'
        summary=json.loads((run/'summary.json').read_text());assert summary['completed'],directory
        predictions=[json.loads(line) for line in (run/'predictions.jsonl').read_text().splitlines()]
        assert len(predictions)==summary.get('predictions',summary.get('measurements',summary.get('rows'))),directory+' row count'
    run=root/'comparison-v1'
    if run.exists():
        config=json.loads((run/'config.json').read_text())
        receipt=json.loads((run/'embedding-receipt.json').read_text())
        assert digest(run/'dataset.json')==receipt['corpus_sha256']==config['dataset_sha256']
        assert digest(run/'embeddings.npz')==receipt['embeddings_sha256']
        assert json.loads((run/'dataset-runtime.json').read_text())==json.loads((run/'dataset.json').read_text())
        summary=json.loads((run/'summary.json').read_text())
        frozen=summary['frozen_constants'];assert frozen['before']==frozen['after'] and frozen['unchanged']
    for manifest_path in root.glob('comparison*/memory/**/manifest.json'):
        manifest=json.loads(manifest_path.read_text())
        assert digest(manifest_path.parent/'updates.safetensors')==manifest['updates_sha256'],str(manifest_path)
    print(json.dumps(dict(verified_artifacts=len(actual),checksums=True,comparison_sources=True,checkpoint_hashes=True)))


if __name__=='__main__':main()
