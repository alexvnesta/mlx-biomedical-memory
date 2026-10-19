"""Portable provenance verification for the trained memory-use experiment."""

import hashlib
import json
from pathlib import Path
import runpy
import struct
import re


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
    dataset=json.loads((root/'decisive-v1/dataset.json').read_text())
    base={c['id']:c for split in ['iid','structure'] for c in dataset['splits'][split]}
    base.update({c['id']:c for c in dataset['interventions']})
    for pilot in sorted(root.glob('decisive-pilot-*')):
        if not (pilot/'config.json').exists():continue
        cfg=json.loads((pilot/'config.json').read_text())
        script='train_decisive_reader.py' if (pilot/'capacity.json').exists() else 'evaluate_decisive_reader.py'
        assert source_hash(pilot,script)==cfg['source_sha256']
        assert cfg['dataset_sha256']==config['dataset_sha256']
        if (pilot/'summary.json').exists():assert json.loads((pilot/'summary.json').read_text())['completed']
        else:assert json.loads((pilot/'stopped.json').read_text())['stopped']
        if (pilot/'predictions.jsonl').exists():
            semantics=runpy.run_path(str(pilot/'source/decisive_data.py'))
            for line in (pilot/'predictions.jsonl').read_text().splitlines():
                row=json.loads(line);case=dict(kind=row['kind'],query=row['query'],records=row['records'],record_id=row['record_id'])
                assert all(row[k]==v for k,v in semantics['grade'](case,row['parsed']).items())
    evaluated=0;trained_evaluated=0
    for evaluation in sorted(root.glob('decisive-*-test')):
        if not re.fullmatch(r'decisive-(?:text|memory)-120[12](?:-trained)?-test',evaluation.name):continue
        is_trained=evaluation.name.endswith('-trained-test')
        cfg=json.loads((evaluation/'config.json').read_text());done=json.loads((evaluation/'summary.json').read_text())
        assert done['completed'] and done['frozen_constants']['unchanged']
        assert done['frozen_constants']['before']==summary['frozen_constants']['before']
        assert source_hash(evaluation,'evaluate_decisive_reader.py')==cfg['source_sha256']
        assert digest(evaluation/'cache_decisive_features.py')==cfg['dependency_sha256']
        assert cfg['dataset_sha256']==config['dataset_sha256'] and not cfg['development']
        training=root/evaluation.name.removesuffix('-test').removesuffix('-trained')
        checkpoint_name='trained-selected.safetensors' if is_trained else 'selected.safetensors'
        trained=json.loads((training/'config.json').read_text());finished=json.loads((training/'summary.json').read_text())
        assert finished['completed'] and finished['frozen_constants']['unchanged']
        assert source_hash(training,'train_decisive_reader.py')==trained['source_sha256']
        assert digest(training/'protocol.md')==trained['protocol_sha256']
        assert digest(training/'cache_decisive_features.py')==trained['dependency_sha256']
        assert digest(features/'manifest.jsonl')==trained['feature_manifest_sha256']
        assert digest(training/checkpoint_name)==cfg['checkpoint_sha256']
        selection=json.loads((training/('trained-only-selection.json' if is_trained else 'generated-selection.json')).read_text())
        if is_trained:assert all(c['step']>0 for c in selection['candidates'])
        assert selection['selected']['checkpoint_sha256']==cfg['checkpoint_sha256']
        assert selection['selected']==max(selection['candidates'],key=lambda c:(c['reasoning_proofs'],c['atomic_correct'],-c['weighted_dev_nll']))
        for candidate in selection['candidates']:
            development=root/Path(candidate['evaluation']).name
            devcfg=json.loads((development/'config.json').read_text())
            assert devcfg['development'] and devcfg['checkpoint_sha256']==candidate['checkpoint_sha256']
            devrows=[json.loads(x) for x in (development/'predictions.jsonl').read_text().splitlines()]
            assert len(devrows)==72
            assert sum(r['proof_valid'] for r in devrows if r['kind']=='reason')==candidate['reasoning_proofs']
            assert sum(r['correct'] for r in devrows if r['kind']=='atomic')==candidate['atomic_correct']
        with (training/checkpoint_name).open('rb') as stream:
            length=struct.unpack('<Q',stream.read(8))[0];header=json.loads(stream.read(length))
        tensors=[v for k,v in header.items() if k!='__metadata__']
        import math
        assert all(v['dtype']=='BF16' for v in tensors)
        assert sum(math.prod(v['shape']) for v in tensors)==finished['trainable_parameters']
        semantics=runpy.run_path(str(evaluation/'source/decisive_data.py'))
        predictions=[json.loads(x) for x in (evaluation/'predictions.jsonl').read_text().splitlines()]
        assert len(predictions)==done['rows']==(512 if cfg['mode']=='memory' else 448)
        assert len({(r['stage'],r['case_id']) for r in predictions})==len(predictions)
        for row in predictions:
            case=dict(id=row['case_id'],kind=row['kind'],query=row['query'],records=row['records'],record_id=row['record_id'])
            reference=base.get(row['case_id']) or base[row['parent_id']]
            assert row['query']==reference['query']
            if row['stage']=='binding':
                without_ids=lambda rs:sorted(json.dumps({k:v for k,v in r.items() if k!='id'},sort_keys=True) for r in rs)
                assert without_ids(row['records'])==without_ids(reference['records'])
            elif row['stage']=='order':
                assert sorted(row['records'],key=lambda r:r['id'])==sorted(reference['records'],key=lambda r:r['id'])
            else:assert row['records']==reference['records']
            result=semantics['grade'](case,row['parsed'])
            assert all(row[k]==v for k,v in result.items()),(evaluation.name,row['case_id'])
            expected=semantics['atomic_solution'](case) if row['kind']=='atomic' else semantics['solution'](case)
            assert row['expected']==expected
        if is_trained:trained_evaluated+=len(predictions)
        else:evaluated+=len(predictions)
    supplement=json.loads((root/'decisive-v1/depth-only.json').read_text())
    depth_by_id={c['id']:c for c in supplement['cases']};depth_evaluated=0
    for directory in sorted(root.glob('decisive-*-depth-test')):
        is_trained=directory.name.endswith('-trained-depth-test')
        cfg=json.loads((directory/'config.json').read_text());done=json.loads((directory/'summary.json').read_text())
        assert done['completed'] and done['rows']==96 and not cfg['development']
        assert done['frozen_constants']['unchanged']
        assert source_hash(directory,'evaluate_decisive_reader.py')==cfg['source_sha256']
        assert cfg['supplemental_sha256']==digest(root/'decisive-v1/depth-only.json')
        assert digest(root/directory.name.removesuffix('-depth-test').removesuffix('-trained')/('trained-selected.safetensors' if is_trained else 'selected.safetensors'))==cfg['checkpoint_sha256']
        semantics=runpy.run_path(str(directory/'source/decisive_data.py'))
        depth_rows=[json.loads(x) for x in (directory/'predictions.jsonl').read_text().splitlines()]
        assert len(depth_rows)==len({r['case_id'] for r in depth_rows})==96
        for row in depth_rows:
            case=depth_by_id[row['case_id']]
            assert row['stage']=='depth_only' and row['records']==case['records'] and row['query']==case['query']
            assert all(row[k]==v for k,v in semantics['grade'](case,row['parsed']).items())
        depth_evaluated+=len(depth_rows)
    audit_cases={c['id']:c for c in json.loads((root/'decisive-v1/training-audit.json').read_text())['cases']};audited=0
    for directory in root.glob('decisive-memory-120[12]-training-audit'):
        cfg=json.loads((directory/'config.json').read_text());done=json.loads((directory/'summary.json').read_text())
        assert done['completed'] and done['rows']==48 and not cfg['development']
        assert cfg['supplemental_sha256']==digest(root/'decisive-v1/training-audit.json')
        assert source_hash(directory,'evaluate_decisive_reader.py')==cfg['source_sha256']
        assert digest(root/directory.name.removesuffix('-training-audit')/'trained-selected.safetensors')==cfg['checkpoint_sha256']
        semantics=runpy.run_path(str(directory/'source/decisive_data.py'))
        audit_rows=[json.loads(x) for x in (directory/'predictions.jsonl').read_text().splitlines()]
        assert len(audit_rows)==len({r['case_id'] for r in audit_rows})==48
        for row in audit_rows:
            case=audit_cases[row['case_id']]
            assert row['records']==case['records'] and row['condition']==case['condition']
            assert all(row[k]==v for k,v in semantics['grade'](case,row['parsed']).items())
        audited+=len(audit_rows)
    if evaluated:
        assert audited==96
        assert depth_evaluated==576
        assert trained_evaluated==1024
        assert evaluated==1920,'Both registered seeds and both readers must be published together'
        latency=root/'decisive-latency'
        latency_config=json.loads((latency/'config.json').read_text());latency_done=json.loads((latency/'summary.json').read_text())
        assert latency_done['completed'] and latency_done['rows']==96
        assert source_hash(latency,'benchmark_decisive_latency.py')==latency_config['source_sha256']
        measurements=[json.loads(x) for x in (latency/'predictions.jsonl').read_text().splitlines()]
        assert len(measurements)==96 and all(r['generated_tokens']==32 for r in measurements)
        assert len({(r['mode'],r['repeat'],r['case_id'],r['prefix_cached']) for r in measurements})==96
        for mode in ['text','memory']:
            assert digest(root/f'decisive-{mode}-1201'/('trained-selected.safetensors' if mode=='memory' else 'selected.safetensors'))==latency_done['checkpoint_sha256'][mode]
        graph=root/'decisive-graph-reference';reference=json.loads((graph/'summary.json').read_text())
        assert reference['completed'] and reference['rows']==352
        assert digest(graph/'source.py')==reference['source_sha256']
        assert digest(graph/'decisive_data.py')==reference['semantics_sha256']
        graph_rows=[json.loads(x) for x in (graph/'predictions.jsonl').read_text().splitlines()]
        assert len(graph_rows)==352 and all(r['proof_valid'] for r in graph_rows)
        depth_graph=json.loads((graph/'depth-summary.json').read_text())
        assert depth_graph['completed'] and depth_graph['rows']==96
        assert depth_graph['dataset_sha256']==digest(root/'decisive-v1/depth-only.json')
        assert depth_graph['source_sha256']==digest(graph/'depth-source.py')
        semantics=runpy.run_path(str(graph/'decisive_data.py'))
        for row in [json.loads(x) for x in (graph/'depth-predictions.jsonl').read_text().splitlines()]:
            assert semantics['grade'](depth_by_id[row['case_id']],row['parsed'])['proof_valid']
        for row in graph_rows:assert semantics['grade'](base[row['case_id']],row['parsed'])['proof_valid']

    print(json.dumps(dict(decisive_feature_sources=True,causal_parity=True,matched_targets=True,no_test_prefix_states=True,regraded_heldout=evaluated,trained_diagnostic_outcomes=trained_evaluated,depth_only_outcomes=depth_evaluated,training_audit_outcomes=audited)))


if __name__=='__main__':verify()
