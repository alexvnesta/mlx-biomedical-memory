"""Fixed-path sign truth tables with irrelevant-edge interventions."""
import copy
import itertools
import math

from engram_poc.decisive_data import world,solution
from engram_poc.readout_data import pair_split


def build():
    rows=[]
    for depth in [1,2,3,4]:
        for index in range(8):
            identifier=f'oracle-depth{depth}-world{index}'
            for attempt in range(1000):
                case=world(identifier,71000000+depth*100000+index*1000+attempt,depth,'UP',10)
                if pair_split(case['query']['root'],case['query']['target'])=='test':break
            else:raise RuntimeError('No oracle world found')
            paths=solution(case)['paths'];assert len(paths)==1
            path=paths[0]['edges'];assert len(path)==depth
            irrelevant=next(r['id'] for r in case['records'] if r['id'] not in path)
            for bits in itertools.product([-1,1],repeat=depth):
                for nuisance in [False,True]:
                    changed=copy.deepcopy(case);byid={r['id']:r for r in changed['records']}
                    for rid,sign in zip(path,bits):byid[rid]['sign']=sign
                    if nuisance:byid[irrelevant]['sign']*=-1
                    expected='UP' if math.prod(bits)==1 else 'DOWN'
                    assert solution(changed)['answer']==expected
                    changed['id']=identifier+'-'+''.join('p' if b==1 else 'm' for b in bits)+('-nuisance' if nuisance else '')
                    rows.append(dict(world_id=identifier,case=changed,path_ids=path,irrelevant_id=irrelevant,
                        nuisance=nuisance,expected_sign=expected))
    assert len(rows)==480
    return rows
