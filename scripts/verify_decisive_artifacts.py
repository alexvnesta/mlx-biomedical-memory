"""Portable provenance verification for the trained memory-use experiment."""

import hashlib
import json
from pathlib import Path


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hash(directory,script):
    root=Path(directory)/'source';h=hashlib.sha256()
    for p in [root/script,*sorted(p for p in root.glob('*.py') if p.name!=script)]:
        h.update(p.name.encode());h.update(p.read_bytes())
    return h.hexdigest()


def verify():
    root=Path('artifacts');features=root/'decisive-features-v1'
    if not features.exists():return
    config=json.loads((features/'config.json').read_text());summary=json.loads((features/'summary.json').read_text())
    assert summary['completed'] and summary['features']==1168
    assert source_hash(features,'cache_decisive_features.py')==config['source_sha256']
    assert digest(root/'decisive-v1/dataset.json')==config['dataset_sha256']
    assert summary['frozen_constants']['unchanged'] and summary['frozen_constants']['before']==summary['frozen_constants']['after']
    assert all(row['max_error']==0 for row in summary['parity'])
    rows=[json.loads(s) for s in (features/'manifest.jsonl').read_text().splitlines()]
    assert len(rows)==1168 and len({(r['id'],r['mode']) for r in rows})==1168
    assert {r['split'] for r in rows}=={'train','dev'},'Held-out target states leaked into feature cache'
    by_id={}
    for row in rows:
        if row['id'] in by_id:assert row['target']==by_id[row['id']]['target']
        by_id[row['id']]=row
    print(json.dumps(dict(decisive_feature_sources=True,causal_parity=True,matched_targets=True,no_test_prefix_states=True)))


if __name__=='__main__':verify()
