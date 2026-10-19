from collections import Counter

from scripts.prepare_composition_experiment import build
from engram_poc.decisive_data import solution,grade
from engram_poc.readout_data import pair_split


def test_composition_queries_are_disjoint_and_gold_depth_is_real():
    corpus=build();pairs={};identities=[]
    for split,partition in [('train','train'),('dev','dev'),('iid','test'),('structure','test')]:
        reason=[c for c in corpus['splits'][split] if c['kind']=='reason']
        pairs[split]={(c['query']['root'],c['query']['target']) for c in reason}
        labels=Counter(solution(c)['answer'] for c in reason)
        assert len(set(labels.values()))==1 and len(labels)==4
        for case in corpus['splits'][split]:
            assert len(case['records'])==10 and {r['id'] for r in case['records']}=={f'F{i}' for i in range(10)}
            identities.append(case['namespace'])
        for case in reason:
            assert pair_split(case['query']['root'],case['query']['target'])==partition
            gold=solution(case);assert grade(case,gold)['proof_valid']
            assert all(len(p['edges'])==case['depth'] for p in gold['paths'])
    assert len(identities)==len(set(identities))
    assert not pairs['train'] & (pairs['dev']|pairs['iid']|pairs['structure'])
    assert not pairs['dev'] & (pairs['iid']|pairs['structure'])
    byid={c['id']:c for split in ['iid','structure'] for c in corpus['splits'][split]}
    for case in corpus['interventions']:assert solution(case)['answer']!=solution(byid[case['parent_id']])['answer']
    for probe in corpus['support_readouts']:
        original=byid[probe['parent_id']]
        assert probe['records']==original['records']
        assert probe['record_id'] in {i for p in solution(original)['paths'] for i in p['edges']}
