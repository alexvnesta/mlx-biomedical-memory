"""Portable verification of published checksums and comparison provenance."""

import hashlib
import json
from pathlib import Path
import runpy
from verify_decisive_artifacts import verify as verify_decisive


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
    fit=root/'relational-fit-v1'
    if fit.exists():
        native_hash=None
        directories=[('relational-fit-v1','fit_relational_memory.py'),
            *((p.name,'run_relational.py') for p in sorted(root.glob('relational-dev-v*')) if p.is_dir()),
            ('relational-v1','run_relational.py'),('relational-transfer-v1','diagnose_scoped_transfer.py'),
            ('relational-boundary-v1','diagnose_value_boundary.py')]
        for directory,script in directories:
            run=root/directory
            if not run.exists():continue
            config=json.loads((run/'config.json').read_text())
            source_hash=hashlib.sha256()
            for path in [run/'source'/script,*sorted(p for p in (run/'source').glob('*.py') if p.name!=script)]:
                source_hash.update(path.name.encode());source_hash.update(path.read_bytes())
            assert source_hash.hexdigest()==config['source_sha256'],directory+' source'
            assert digest(run/'dataset.json')==config['dataset_sha256'],directory+' dataset'
            summary=json.loads((run/'summary.json').read_text());assert summary['completed'],directory
            if 'frozen_constants' in summary:
                frozen=summary['frozen_constants']
                assert frozen['before']==frozen['after'] and frozen['unchanged'],directory+' frozen backbone'
                if native_hash is None:native_hash=frozen['before']
                assert native_hash==frozen['before'],directory+' native checkpoint identity'
            if directory!='relational-fit-v1':
                predictions=[json.loads(line) for line in (run/'predictions.jsonl').read_text().splitlines()]
                assert len(predictions)==summary['rows'],directory+' row count'
                assert config['checkpoint_sha256']==digest(fit/'memory/scoped.safetensors')
                assert config['dataset_sha256']==digest(fit/'dataset.json')
                cells={(r['stage'],r['case_id'],r['method'],r.get('form','')) for r in predictions}
                assert len(cells)==len(predictions),directory+' duplicate cell'
                if directory=='relational-transfer-v1':
                    assert len(predictions)==180
                    assert digest(run/'run_relational.py')==config['dependency_sha256']
                    assert all(r['scoped_probe_exact'] and r['scoped_probe_max_error']==0 for r in predictions)
                if directory=='relational-boundary-v1':
                    assert len(predictions)==32
                    assert digest(run/'run_relational.py')==config['dependency_sha256']
                    assert {r['case_id'] for r in predictions}=={'atomic-0','atomic-1','atomic-2','atomic-3'}
                if directory=='relational-v1':
                    assert len(predictions)==972
                    receipt=json.loads((run/'embedding-receipt.json').read_text())
                    assert receipt['embeddings_sha256']==digest(run/'document-embeddings.npz')
                    arax=json.loads((run/'arax-audit.json').read_text())
                    assert arax['dataset_sha256']==config['dataset_sha256']
                    assert arax['audit_source_sha256']==digest(run/'arax-audit.py')
                    assert sum(r['verified_records'] for r in arax['projections'])==36
                    semantics=runpy.run_path(str(run/'source/relational.py'))
                    corpus=json.loads((run/'dataset.json').read_text())
                    cases={c['id']:c for c in corpus['cases']}
                    interventions={v['case']['id']:v for v in corpus['interventions']}
                    records={r['id']:r for r in corpus['records']}
                    for row in predictions:
                        if row['stage'] not in ['initial','intervention']:continue
                        current=dict(records)
                        if row['stage']=='intervention':
                            intervention=interventions[row['case_id']];case=intervention['case']
                            current.update({r['id']:r for r in intervention['replacement_records']})
                        else:case=cases[row['case_id']]
                        replay=semantics['grade'](case,row['answer'],row['evidence'],row['retrieved_ids'],
                            certificates=row['completeness_certificates'],records=[current[i] for i in row['retrieved_ids']])
                        for field,value in replay.items():assert row[field]==value,(row['case_id'],row['method'],field)
                        if row['method'].startswith('agent_'):
                            graph=semantics['EvidenceGraph'](list(current.values()))
                            returned=set()
                            for certificate in row['completeness_certificates']:
                                result=graph.lookup(certificate['entity'],certificate['relation'],
                                    certificate['context'],certificate['direction'])
                                assert result['complete']==certificate['complete']
                                assert [r['id'] for r in result['records']]==certificate['record_ids']
                                returned.update(certificate['record_ids'])
                            assert returned==set(row['retrieved_ids'])
                            graph.db.close()
        manifest=json.loads((fit/'memory/manifest.json').read_text())
        assert manifest['vectors_sha256']==digest(fit/'memory/scoped.safetensors')
        assert len(manifest['keys'])==len(set(manifest['keys']))==30
        assert sorted(manifest['permutation'])==list(range(31))
        ngram=json.loads((fit/'memory/ngram/manifest.json').read_text())
        assert ngram['updates_sha256']==digest(fit/'memory/ngram/updates.safetensors')
        boundary=root/'relational-boundary-v1/memory'
        if boundary.exists():
            manifest=json.loads((boundary/'manifest.json').read_text())
            assert manifest['vectors_sha256']==digest(boundary/'scoped.safetensors')
            ngram=json.loads((boundary/'ngram/manifest.json').read_text())
            assert ngram['updates_sha256']==digest(boundary/'ngram/updates.safetensors')
    print(json.dumps(dict(verified_artifacts=len(actual),checksums=True,comparison_sources=True,
        relational_sources=fit.exists(),checkpoint_hashes=True)))
    verify_decisive()


if __name__=='__main__':main()
