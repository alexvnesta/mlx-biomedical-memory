"""Portable replay of readout repair grading, partitions and provenance."""
import hashlib
import json
import re
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from engram_poc.decisive_data import atomic_solution
from engram_poc.readout_data import make_case, pair_split


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def decoded(text):
    for i,char in enumerate(text):
        if char!='{':continue
        try:value,_=json.JSONDecoder().raw_decode(text[i:])
        except json.JSONDecodeError:continue
        if isinstance(value,dict) and 'record' in value:return value
    return None


def source_hash(run,script):
    h=hashlib.sha256()
    for p in [run/'source'/script,*sorted(p for p in (run/'source').glob('*.py') if p.name!=script)]:
        h.update(p.name.encode());h.update(p.read_bytes())
    return h.hexdigest()


def verify(root=Path('artifacts')):
    diagnosis=root/'readout-diagnosis-v1'
    if diagnosis.exists():
        rows=[json.loads(x) for x in (diagnosis/'predictions.jsonl').read_text().splitlines()]
        assert len(rows)==16
        assert all(r['cached_direct_max_error']==0 and r['prefix_incremental_max_error']==0 for r in rows)
        bycase={r['case_id']:r for r in rows if r['precision']=='fp32'}
        for r in rows:
            assert r['parsed']==decoded(r['generation'])
            assert r['correct']==(r['parsed']==r['expected'])
            if r['precision']=='bf16':assert r['parsed']==bycase[r['case_id']]['parsed']
    evaluated=0
    for run in sorted(root.glob('readout-repair-test-*')):
        cfg=json.loads((run/'config.json').read_text());summary=json.loads((run/'summary.json').read_text())
        assert cfg['source_sha256']==source_hash(run,'evaluate_readout_repair.py')
        assert cfg['case_sha256']==sha(run/'cases.json')
        training=root/Path(cfg['training']).name
        assert cfg['checkpoint_sha256']==sha(training/cfg['checkpoint_name'])
        selection=json.loads((training/'selection.json').read_text())
        train_cfg=json.loads((training/'config.json').read_text())
        assert train_cfg['source_sha256']==source_hash(training,'train_readout_curriculum.py')
        assert (run/'recipe.py').read_bytes()==(training/'source/train_readout_curriculum.py').read_bytes()
        dev=[json.loads(x) for x in (training/'development.jsonl').read_text().splitlines()]
        for row in dev:
            assert row['expected']==atomic_solution(row['case'])
            assert row['correct']==(row['parsed']==row['expected'])
            assert all(pair_split(r['subject'],r['object'])=='dev' for r in row['case']['records'])
        chosen=[r for r in dev if r['step']==selection['step']]
        assert sum(r['correct'] for r in chosen)==selection['correct'] and len(chosen)==selection['total']
        for line in (training/'training.jsonl').read_text().splitlines():
            row=json.loads(line)
            for draw in range(2):
                case=make_case(train_cfg['seed']*1000000+row['step']*2+draw,row['count'])
                assert all(pair_split(r['subject'],r['object'])=='train' for r in case['records'])
        cases={r['case']['id']:r for r in json.loads((run/'cases.json').read_text())}
        rows=[json.loads(x) for x in (run/'predictions.jsonl').read_text().splitlines()]
        assert len(rows)==len(cases)==summary['rows']==896
        totals={};byid={r['case_id']:r for r in rows}
        for row in rows:
            spec=cases[row['case_id']];case=spec['case'];condition=spec['condition']
            assert row['condition']==condition and row['parent_id']==spec['parent_id']
            assert row['records']==case['records'] and row['record_id']==case['record_id']
            assert row['injected_records']==case.get('injected_records',case['records'])
            expected=atomic_solution(case);injected=atomic_solution(dict(case,records=row['injected_records']))
            assert row['expected']==expected and row['injected_expected']==injected
            assert row['parsed']==decoded(row['generation'])
            assert row['correct']==(row['parsed']==expected)
            assert row['injected_correct']==(row['parsed']==injected)
            assert row['field_correct']=={k:(row['parsed'] or {}).get(k)==v for k,v in expected.items()}
            if condition=='normal':
                assert all(pair_split(r['subject'],r['object'])=='test' for r in case['records'])
            assert not re.search(r'\b(?:N\d+|S\d+|UP|DOWN|human|mouse|lof|gof)\b',row['prompt'])
            if row['parent_id']:
                parent=byid[row['parent_id']]
                if condition!='order':assert row['prompt']==parent['prompt']
                if condition.startswith('change_'):
                    field=condition.removeprefix('change_')
                    differences=[(r['id'],k) for r,s in zip(case['records'],parent['records']) for k in r if r[k]!=s[k]]
                    assert differences==[(row['record_id'],field)]
                elif condition=='order':assert sorted(case['records'],key=lambda r:r['id'])==sorted(parent['records'],key=lambda r:r['id'])
                elif condition=='binding':
                    for i,r in enumerate(case['records']):
                        assert r['id']==parent['records'][i]['id']
                        assert {k:v for k,v in r.items() if k!='id'}=={k:v for k,v in parent['records'][(i+1)%10].items() if k!='id'}
            aggregate=totals.setdefault(condition,dict(correct=0,total=0,injected_correct=0,field_correct={k:0 for k in expected}))
            aggregate['total']+=1;aggregate['correct']+=row['correct'];aggregate['injected_correct']+=row['injected_correct']
            for k in expected:aggregate['field_correct'][k]+=row['field_correct'][k]
        assert totals==summary['conditions']
        assert summary['frozen_constants']['before']==summary['frozen_constants']['after']=='dc059288f9bd691462722252054f8e9707f8dc700339c8731fca98f454839edb'
        evaluated+=len(rows)
    print(json.dumps(dict(readout_diagnostic_replayed=diagnosis.exists(),readout_repair_outcomes=evaluated)))


if __name__=='__main__':verify()
