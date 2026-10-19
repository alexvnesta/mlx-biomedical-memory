from collections import Counter

from engram_poc.readout_data import make_case, pair_split, noisy_prefix
from engram_poc.decisive_data import atomic_solution


def test_unseen_pairs_keep_every_individual_entity_in_all_splits():
    pairs={split:{(s,o) for s in range(16) for o in range(16) if pair_split(s,o)==split}
        for split in ['train','dev','test']}
    assert list(map(len,pairs.values()))==[192,32,32]
    assert not pairs['train'] & (pairs['dev']|pairs['test'])
    assert not pairs['dev'] & pairs['test']
    for values in pairs.values():
        assert {s for s,o in values}==set(range(16))
        assert {o for s,o in values}==set(range(16))


def test_bank_partition_includes_decoys_and_every_field_varies():
    for split in ['train','dev','test']:
        observed={k:set() for k in ['id','subject','object','sign','species','variant','source']}
        positions=Counter()
        for seed in range(64):
            case=make_case(seed,10,split)
            assert len({r['id'] for r in case['records']})==10
            assert all(pair_split(r['subject'],r['object'])==split for r in case['records'])
            assert atomic_solution(case)['record']==case['record_id']
            positions[next(i for i,r in enumerate(case['records']) if r['id']==case['record_id'])]+=1
            for r in case['records']:
                for k in observed:observed[k].add(r[k])
        assert all(len(v)>1 for v in observed.values())
        assert set(positions)==set(range(10))


def test_prefix_noise_preserves_truth_and_token_positions():
    import json
    values=atomic_solution(make_case(5,1));original=dict(values)
    encode=lambda s:list(s.encode())
    changed,tokens=noisy_prefix(values,encode,5,1.)
    assert values==original and changed['record']==values['record']
    assert len(tokens)==len(encode(json.dumps(values,separators=(',',':'))))
    assert all(changed[k]!=values[k] for k in ['subject','object','source'])
    unchanged,tokens=noisy_prefix(values,encode,5,0.)
    assert unchanged==values
