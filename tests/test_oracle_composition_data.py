from collections import Counter,defaultdict
import math

from scripts.oracle_composition_data import build
from engram_poc.decisive_data import solution


def test_oracle_sweeps_every_sign_combination_and_irrelevant_flip():
    rows=build();groups=defaultdict(list)
    for row in rows:groups[row['world_id']].append(row)
    assert len(rows)==480 and len(groups)==32
    for group in groups.values():
        first=group[0];depth=first['case']['depth'];path=first['path_ids']
        assert len(group)==2**(depth+1) and len(path)==depth
        combinations=Counter()
        for row in group:
            assert row['case']['query']==first['case']['query'] and row['path_ids']==path
            records={r['id']:r for r in row['case']['records']}
            signs=tuple(records[i]['sign'] for i in path);combinations[signs]+=1
            assert row['expected_sign']==('UP' if math.prod(signs)==1 else 'DOWN')
            assert solution(row['case'])['answer']==row['expected_sign']
        assert len(combinations)==2**depth and set(combinations.values())=={2}
        assert Counter(r['expected_sign'] for r in group)=={'UP':len(group)//2,'DOWN':len(group)//2}
