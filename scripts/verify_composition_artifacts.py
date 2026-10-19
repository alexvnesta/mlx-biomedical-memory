"""Portable replay of the readout-gated graph comparison."""
import ast
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import re
import runpy
import struct
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from verify_readout_artifacts import source_hash


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parse(text,kind):
    for i,c in enumerate(text):
        if c!='{':continue
        try:value,_=json.JSONDecoder().raw_decode(text[i:])
        except json.JSONDecodeError:continue
        if isinstance(value,dict) and ('record' in value if kind=='atomic' else 'answer' in value):return value
    return None


def tensors(path,prefix=''):
    data=Path(path).read_bytes();n=struct.unpack('<Q',data[:8])[0];header=json.loads(data[8:8+n]);start=8+n
    return {k:({a:b for a,b in v.items() if a!='data_offsets'},data[start+v['data_offsets'][0]:start+v['data_offsets'][1]]) for k,v in header.items() if k!='__metadata__' and k.startswith(prefix)}


def verify(root=Path('artifacts')):
    features=root/'composition-features-v1'
    if not features.exists():return
    cfg=json.loads((features/'config.json').read_text());done=json.loads((features/'summary.json').read_text())
    assert source_hash(features,'cache_composition_features.py')==cfg['source_sha256']
    assert sha(features/'dataset.json')==cfg['dataset_sha256']
    assert sha(features/'cache_decisive_features.py')==cfg['dependency_sha256']
    assert done['completed'] and done['features']==1760
    assert done['frozen_constants']['before']==done['frozen_constants']['after']=='dc059288f9bd691462722252054f8e9707f8dc700339c8731fca98f454839edb'
    assert all(r['max_error']==0 for r in done['parity'])
    semantics=runpy.run_path(str(features/'source/decisive_data.py'))
    data_helpers=runpy.run_path(str(features/'source/readout_data.py'))
    producer=runpy.run_path(str(features/'prepare_composition_experiment.py'))
    generator=producer['build'];generator.__globals__.update({k:semantics[k] for k in ['world','solution','LABELS']})
    generator.__globals__.update({k:data_helpers[k] for k in ['make_case','pair_split']})
    corpus=json.loads((features/'dataset.json').read_text());expected=generator()
    assert {k:v for k,v in corpus.items() if k!='readout_prerequisite'}==expected
    prerequisite=corpus['readout_prerequisite']
    assert prerequisite['summary_sha256']==sha(root/'readout-repair-test-v5/summary.json')
    warm=root/'readout-curriculum-v5/selected.safetensors'
    assert prerequisite['checkpoint_sha256']==sha(warm)=='3d913d797cf7209590c88336a014ca44d6a6197fe422f2ecc92d01ef9f3bb3c1'
    warm_memory=tensors(warm,'memory.')
    manifest=[json.loads(x) for x in (features/'manifest.jsonl').read_text().splitlines()]
    assert len(manifest)==1760 and {r['split'] for r in manifest}=={'train','dev'}
    examples={c['id']:c for s in ['train','dev'] for c in corpus['splits'][s]};pairs={}
    for r in manifest:
        assert r['id'] in examples and r['kind']==examples[r['id']]['kind']
        assert r['fields']==semantics['fields'](examples[r['id']]['records'])
        assert len(r['input_target'])+1==len(r['target'])
        assert r['shape']==[1,r['loss_start']+len(r['target']),3072]
        if r['split']=='dev' or r['kind']=='reason':assert r['input_target']==r['target'][:-1]
        assert len(r['positions'])==(10 if r['mode']=='memory' else 0)
        pairs.setdefault(r['id'],{})[r['mode']]=r
    assert set(pairs)==set(examples)
    for values in pairs.values():
        assert set(values)=={'memory','text'}
        assert values['memory']['target']==values['text']['target'] and values['memory']['input_target']==values['text']['input_target']
    evaluations={};count=0
    for run in sorted(root.glob('composition-*')):
        if not re.fullmatch(r'composition-(?:memory|text|joint)-240[12](?:-dev-.+|-test)?',run.name):continue
        if not (run/'config.json').exists():continue
        config=json.loads((run/'config.json').read_text());summary=json.loads((run/'summary.json').read_text())
        training_run=not (run/'predictions.jsonl').exists()
        script='train_decisive_reader.py' if training_run else 'evaluate_decisive_reader.py'
        assert config['source_sha256']==source_hash(run,script)
        assert config['dataset_sha256']==cfg['dataset_sha256'] and summary['completed']
        assert sha(run/'cache_decisive_features.py')==config['dependency_sha256']
        assert summary['frozen_constants']['unchanged'] and summary['frozen_constants']['before']==summary['frozen_constants']['after']
        if training_run:
            assert summary['frozen_constants']['before']=='9d06632f740f45de839147fed0a19e5c14263ceaea962868bc37128fe15f60d6'
            assert config['feature_manifest_sha256']==sha(features/'manifest.jsonl')
            assert config['protocol_sha256']==sha(run/'protocol.md')
            capacity=json.loads((run/'capacity.json').read_text())
            assert capacity['trainable_parameters']==(5914624 if config['mode']=='text' else 2957312 if config['frozen_codec'] else 5719809)
            if config['frozen_codec']:
                assert summary['frozen_codec']['before']==summary['frozen_codec']['after'] and summary['frozen_codec']['unchanged']
            for checkpoint in run.glob('*.safetensors'):
                arrays=tensors(checkpoint)
                assert all(v[0]['dtype']=='BF16' for v in arrays.values())
                assert sum(math.prod(v[0]['shape']) for v in arrays.values())==(5914624 if config['mode']=='text' else 5719809)
                if config['frozen_codec']:assert tensors(checkpoint,'memory.')==warm_memory
                elif config['mode']=='memory' and (checkpoint.name.startswith('final-') or checkpoint.name=='selected.safetensors'):
                    assert tensors(checkpoint,'memory.')!=warm_memory,'Joint adaptation did not change the codec'
            continue
        assert summary['frozen_constants']['before']==done['frozen_constants']['before']
        training=root/re.match(r'(composition-(?:memory|text|joint)-240[12])',run.name).group(1)
        assert sha(training/config['checkpoint_name'])==config['checkpoint_sha256']
        if config['development']:cases=[('development',c,'normal') for c in corpus['splits']['dev']]
        else:
            tree=ast.parse((run/'source/evaluate_decisive_reader.py').read_text())
            fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='variations')
            ns=dict(copy=copy,random=random);exec(compile(ast.Module(body=[fn],type_ignores=[]),'frozen-variations','exec'),ns)
            cases=ns['variations'](corpus)
            assert config['include_support_readouts']
            cases += [('support_readout',c,'normal') for c in corpus['support_readouts']]
            if config['mode']=='text':cases=[x for x in cases if x[0]!='masked']
        case_map={(stage,c['id']):(c,condition) for stage,c,condition in cases}
        rows=[json.loads(x) for x in (run/'predictions.jsonl').read_text().splitlines()]
        assert len(rows)==len(case_map)==summary['rows'] and {(r['stage'],r['case_id']) for r in rows}==set(case_map)
        for r in rows:
            case,condition=case_map[r['stage'],r['case_id']]
            assert r['condition']==condition and r['records']==case['records'] and r['query']==case['query']
            assert r['parsed']==parse(r['generation'],r['kind'])
            truth=semantics['atomic_solution'](case) if r['kind']=='atomic' else semantics['solution'](case)
            assert r['expected']==truth
            assert all(r[k]==v for k,v in semantics['grade'](case,r['parsed']).items())
        evaluations[run.name]=rows;count+=len(rows)
    for run in root.glob('composition-*-240[12]'):
        selection=run/'generated-selection.json'
        if not selection.exists():continue
        receipt=json.loads(selection.read_text());eligible=[]
        for candidate in receipt['candidates']:
            rows=evaluations[Path(candidate['evaluation']).name]
            assert sum(r['proof_valid'] for r in rows if r['kind']=='reason')==candidate['reasoning_proofs']
            assert sum(r['correct'] for r in rows if r['kind']=='atomic')==candidate['atomic_correct']
            assert sha(run/candidate['checkpoint_name'])==candidate['checkpoint_sha256']
            if candidate['step']>0 and candidate['atomic_correct']>=63 and candidate['atomic_total']==64:eligible.append(candidate)
        chosen=max(eligible,key=lambda c:(c['reasoning_proofs'],c['atomic_correct'],-c['weighted_dev_nll'])) if eligible else None
        assert chosen==receipt['selected']
        if chosen:assert sha(run/'selected.safetensors')==chosen['checkpoint_sha256']
    for name in ['composition-v1','composition-joint-v1']:
        run=root/name
        if not run.exists():continue
        config=json.loads((run/'config.json').read_text());assert sha(run/'controller.py')==config['controller_sha256']
        assert config['dataset_sha256']==cfg['dataset_sha256'] and config['feature_manifest_sha256']==sha(features/'manifest.jsonl')
        assert config['warm_start_sha256']==sha(warm)
        if name=='composition-v1':assert sha(run/'protocol.md')==config['protocol_sha256']
        else:
            assert sha(run/'protocol.md')==config['amendment_sha256']
            assert sha(run/'control-controller.py')==config['controller_dependency_sha256']
            assert sha(root/'composition-v1/development-gate.json')==config['control_gate_sha256']
            assert sha(run/'oracle-protocol.md')==config['oracle_protocol_sha256']
            assert sha(run/'retention-amendment.md')==config['retention_amendment_sha256']
            for filename,digest in config['oracle_sources'].items():assert sha(run/'oracle-sources'/filename)==digest
        if (run/'development-gate.json').exists():
            gate=json.loads((run/'development-gate.json').read_text())
            assert gate['passed']==all(item is not None for item in gate['selection'].values())
            for key,item in gate['selection'].items():
                assert item==json.loads((root/('composition-'+key)/'generated-selection.json').read_text())['selected']
                if item is None:
                    assert not (root/('composition-'+key+'-test')).exists()
                    continue
                assert item['atomic_correct']>=63 and item['step']>0
                assert sha(root/('composition-'+key)/'selected.safetensors')==item['checkpoint_sha256']
    oracle=root/'composition-oracle-features-v1';oracle_count=0
    if oracle.exists():
        cfg_o=json.loads((oracle/'config.json').read_text());summary_o=json.loads((oracle/'summary.json').read_text())
        assert cfg_o['source_sha256']==source_hash(oracle,'cache_oracle_composition.py')
        assert cfg_o['producer_sha256']==sha(oracle/'oracle_composition_data.py')
        assert cfg_o['protocol_sha256']==sha(oracle/'protocol.md')
        assert cfg_o['dependency_sha256']==sha(oracle/'cache_decisive_features.py')
        assert cfg_o['cases_sha256']==sha(oracle/'cases.json')
        assert summary_o['completed'] and summary_o['rows']==960 and summary_o['unique_memory_prefixes']==32
        factory=runpy.run_path(str(oracle/'oracle_composition_data.py'))['build']
        factory.__globals__.update({k:semantics[k] for k in ['world','solution']})
        factory.__globals__['pair_split']=data_helpers['pair_split']
        cases=json.loads((oracle/'cases.json').read_text());assert cases==factory()
        byid={r['case']['id']:r for r in cases}
        manifest_o=[json.loads(x) for x in (oracle/'manifest.jsonl').read_text().splitlines()]
        assert len(manifest_o)==960
        memgroups={}
        for row in manifest_o:
            case=byid[row['case_id']]
            assert row['fields']==semantics['fields'](case['case']['records'])
            assert row['path_ids']==case['path_ids'] and row['expected_sign']==case['expected_sign']
            assert row['shape']==[1,row['input_tokens'],3072]
            if row['mode']=='memory':memgroups.setdefault(row['world_id'],set()).add((row['token_sha256'],row['feature_sha256']))
        assert len(memgroups)==32 and all(len(v)==1 for v in memgroups.values())
        signids={k:v[0] for k,v in json.loads((oracle/'sign-token-ids.json').read_text()).items()}
        for run in root.glob('composition-*-240[12]-oracle'):
            config=json.loads((run/'config.json').read_text());summary=json.loads((run/'summary.json').read_text())
            assert config['source_sha256']==source_hash(run,'evaluate_oracle_composition.py')
            assert config['manifest_sha256']==sha(oracle/'manifest.jsonl')
            assert config['dependency_sha256']==sha(run/'cache_decisive_features.py')==cfg_o['dependency_sha256']
            assert config['checkpoint_sha256']==sha(root/Path(config['training']).name/'selected.safetensors')
            assert summary['frozen_constants']['before']==summary['frozen_constants']['after']=='9d06632f740f45de839147fed0a19e5c14263ceaea962868bc37128fe15f60d6'
            rows=[json.loads(x) for x in (run/'predictions.jsonl').read_text().splitlines()]
            conditions=['normal'] if config['mode']=='text' else ['normal','masked']
            assert summary['completed'] and len(rows)==summary['rows']==480*len(conditions)
            assert len({(r['case_id'],r['condition']) for r in rows})==len(rows)
            for row in rows:
                original=byid[row['case_id']]
                assert row['expected_sign']==original['expected_sign'] and row['world_id']==original['world_id']
                assert row['correct']==(row['prediction_token']==signids[row['expected_sign']])
            for condition in conditions:
                for depth in [1,2,3,4]:
                    part=[r for r in rows if r['condition']==condition and r['depth']==depth]
                    worlds={r['world_id'] for r in part};value=summary['results'][condition][str(depth)]
                    assert len(part)==value['total']==8*2**(depth+1) and len(worlds)==value['worlds']==8
                    assert sum(r['correct'] for r in part)==value['correct']
                    assert sum(all(r['correct'] for r in part if r['world_id']==w) for w in worlds)==value['worlds_all_correct']
            oracle_count+=len(rows)
    report=root/'composition-summary.json'
    if report.exists():
        summary=json.loads(report.read_text())
        payload=json.loads(re.search(r'<script id="data" type="application/json">(.*?)</script>',(root/'composition.html').read_text(),re.S)[1])
        assert payload['summary']==summary
        mapped={'text':'text','memory':'frozen','joint':'joint'}
        expected=[]
        for name,rows in evaluations.items():
            if name.endswith('-test'):
                method=mapped[name.split('-')[1]]
                expected += [dict(row,method=method) for row in rows]
        canonical=lambda rows:sorted(json.dumps(r,sort_keys=True) for r in rows)
        assert canonical(payload['rows'])==canonical(expected)
        for method,metrics in summary['metrics'].items():
            batch=[r for r in expected if r['method']==method]
            atomic=[r for r in batch if r['kind']=='atomic' and r['stage'] in ['iid','structure']]
            assert metrics['atomic']==dict(correct=sum(r['correct'] for r in atomic),total=len(atomic))
            for depth in [1,2,3,4]:
                part=[r for r in batch if r['kind']=='reason' and r['stage'] in ['iid','structure'] and r['expected']['answer']!='UNKNOWN' and r['depth']==depth]
                assert metrics['depths'][str(depth)]==dict(correct=sum(r['proof_valid'] for r in part),total=len(part))
        for memory,comparison in summary['matched_seed_comparisons'].items():
            for stage,values in comparison['strata'].items():
                for method,value in values.items():
                    batch=[r for r in expected if r['method']==method and r['seed'] in comparison['seeds'] and r['kind']=='reason' and r['stage']==stage and r['expected']['answer']!='UNKNOWN']
                    assert value==dict(correct=sum(r['proof_valid'] for r in batch),total=len(batch))
    print(json.dumps(dict(composition_features=1760,composition_generations_regraded=count,oracle_decisions_regraded=oracle_count)))


if __name__=='__main__':verify()
