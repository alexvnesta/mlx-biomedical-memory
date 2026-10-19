"""Fresh graph compositions, after freezing a successfully decoded memory."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

from engram_poc.decisive_data import world, solution, LABELS
from engram_poc.readout_data import make_case, pair_split


def build():
    splits={}
    specifications=[('train',384,384,[1,2],61000000,'train'),
        ('dev',48,64,[1,2],62000000,'dev'),('iid',96,64,[1,2],63000000,'test'),
        ('structure',96,64,[3,4],64000000,'test')]
    for split,reason_count,atomic_count,depths,seed,partition in specifications:
        cases=[]
        for i in range(reason_count):
            for attempt in range(1000):
                case=world(f'composition-{split}-{i:04d}',seed+i*1000+attempt,
                    depths[(i//4)%len(depths)],LABELS[i%4],10)
                if pair_split(case['query']['root'],case['query']['target'])==partition:break
            else:raise RuntimeError('Failed to sample query partition')
            cases.append(case)
        for i in range(atomic_count):
            case=make_case(seed+100000000+i,10,partition)
            identifier=f'composition-{split}-atomic-{i:04d}'
            case.update(id=identifier,namespace=identifier,query=None,depth=0)
            cases.append(case)
        splits[split]=cases
    interventions=[]
    for split in ['iid','structure']:
        for case in [c for c in splits[split] if c['kind']=='reason'][:32]:
            label=solution(case)['answer'];changed=None
            for i in range(10):
                for field in ['sign','species','variant']:
                    trial=copy.deepcopy(case);r=trial['records'][i]
                    r[field]=-r[field] if field=='sign' else 1-r[field]
                    if solution(trial)['answer']!=label:
                        trial.update(id=case['id']+'-changed',parent_id=case['id'],changed_record=r['id'],changed_field=field)
                        changed=trial;break
                if changed:break
            assert changed is not None;interventions.append(changed)
    support=[]
    for split in ['iid','structure']:
        selected=[c for c in splits[split] if c['kind']=='reason' and solution(c)['paths']][:16]
        for original in selected:
            for identifier in sorted({i for p in solution(original)['paths'] for i in p['edges']}):
                case=copy.deepcopy(original)
                case.update(id=original['id']+'-support-'+identifier,parent_id=original['id'],kind='atomic',record_id=identifier)
                support.append(case)
    return dict(splits=splits,interventions=interventions,support_readouts=support,
        semantics='Complete synthetic worlds and stipulated sign algebra, not biological causal claims',
        global_entities='World-scoped aliases; every world namespace is new',
        holdout='TRACE query root/target pairs partitioned by (5*root+3*target)%8; train=2..7, dev=0, test=1. Bank edges are unrestricted.',
        note='Fresh atomic retention banks use the same partitions, but reasoning-training bank edges can contain these local alias pairs. No claim of atomic pair exclusion after reasoning training.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('data/processed/composition-v1/dataset.json'))
    p.add_argument('--readout',type=Path,default=Path('runs/readout-repair-test-v5'))
    args=p.parse_args()
    summary=json.loads((args.readout/'summary.json').read_text())
    assert summary['completed'] and summary['gate_passed'],'Readout prerequisite must pass first'
    assert not args.output.exists(),'Preserve registered corpora'
    data=build();data['readout_prerequisite']=dict(summary_sha256=hashlib.sha256((args.readout/'summary.json').read_bytes()).hexdigest(),
        checkpoint_sha256=json.loads((args.readout/'config.json').read_text())['checkpoint_sha256'])
    args.output.parent.mkdir(exist_ok=True,parents=True)
    args.output.write_text(json.dumps(data,indent=2)+'\n')
    print(json.dumps(dict(dataset_sha256=hashlib.sha256(args.output.read_bytes()).hexdigest(),
        cases={k:len(v) for k,v in data['splits'].items()},interventions=len(data['interventions']))))


if __name__=='__main__':main()
