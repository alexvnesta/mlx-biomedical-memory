"""Query-independent fact banks and fresh-world signed-relation tests.

Signs are stipulated algebra, not biological causal claims. Global entity IDs
are (world namespace, local node); local aliases are shared across input modes.
"""

import copy
import json
from pathlib import Path
import random


LABELS=['UP','DOWN','BOTH','UNKNOWN']


def paths(case):
    p=case['query'];rows=case['records'];found={1:[],-1:[]}
    def visit(node,product,seen,edges):
        if node==p['target']:
            found[product].append(edges);return
        for r in rows:
            if r['subject']==node and r['species']==p['species'] and r['variant']==p['variant'] and r['object'] not in seen:
                visit(r['object'],product*r['sign'],seen|{r['object']},edges+[r['id']])
    visit(p['root'],1,{p['root']},[])
    for sign in found:found[sign].sort(key=lambda x:(len(x),x))
    return found


def solution(case):
    found=paths(case);positive=bool(found[1]);negative=bool(found[-1])
    label='BOTH' if positive and negative else 'UP' if positive else 'DOWN' if negative else 'UNKNOWN'
    trace=[dict(edges=found[s][0],sign='UP' if s==1 else 'DOWN') for s in [1,-1] if found[s]]
    evidence=sorted({i for path in trace for i in path['edges']}) if trace else sorted(r['id'] for r in case['records'])
    return dict(paths=trace,answer=label,evidence=evidence)


def grade(case,output):
    if case.get('kind')=='atomic':
        expected=atomic_solution(case)
        return dict(correct=output==expected,proof_valid=output==expected,invalid=not isinstance(output,dict))
    expected=solution(case);obj=output if isinstance(output,dict) else {}
    correct=obj.get('answer')==expected['answer'];rows={r['id']:r for r in case['records']}
    evidence=obj.get('evidence',[]);trace=obj.get('paths',[]);valid=True;observed=set();used=set()
    if not isinstance(evidence,list) or not all(isinstance(x,str) and x in rows for x in evidence):valid=False;evidence=[]
    if not isinstance(trace,list):valid=False;trace=[]
    for item in trace:
        if not isinstance(item,dict):valid=False;continue
        edge_ids=item.get('edges',[]);node=case['query']['root'];product=1;seen={node}
        if not isinstance(edge_ids,list) or not edge_ids:valid=False;continue
        for identifier in edge_ids:
            if not isinstance(identifier,str) or identifier not in rows:valid=False;break
            r=rows[identifier];q=case['query']
            if r['subject']!=node or r['object'] in seen or r['species']!=q['species'] or r['variant']!=q['variant']:
                valid=False;break
            node=r['object'];seen.add(node);product*=r['sign'];used.add(identifier)
        else:
            sign='UP' if product==1 else 'DOWN'
            if node!=case['query']['target'] or item.get('sign')!=sign:valid=False
            else:observed.add(sign)
    label=obj.get('answer')
    required={'UP'} if label=='UP' else {'DOWN'} if label=='DOWN' else {'UP','DOWN'} if label=='BOTH' else set()
    if label=='UNKNOWN':valid=valid and not trace and set(evidence)==set(rows)
    else:valid=valid and observed==required and bool(trace) and used<=set(evidence)
    return dict(correct=correct,proof_valid=bool(correct and valid),invalid=not isinstance(output,dict))


def atomic_solution(case):
    r=next(r for r in case['records'] if r['id']==case['record_id'])
    return dict(record=r['id'],subject=f"N{r['subject']}",object=f"N{r['object']}",
        sign='UP' if r['sign']==1 else 'DOWN',species='human' if r['species']==0 else 'mouse',
        variant='lof' if r['variant']==0 else 'gof',source=f"S{r['source']}")


def fields(records):
    """No query, answer, path search or dataset split is available here."""
    return [[int(r['id'][1:]),r['subject'],r['object'],0 if r['sign']==1 else 1,
        r['species'],r['variant'],r['source']] for r in records]


def world(identifier,seed,depth,label,record_count):
    rng=random.Random(seed);nodes=list(range(16));rng.shuffle(nodes);root,target=nodes[:2];cursor=2
    q=dict(root=root,target=target,species=rng.randrange(2),variant=rng.randrange(2));rows=[]
    def edge(a,b,sign,species=None,variant=None):
        rows.append(dict(subject=a,object=b,sign=sign,species=q['species'] if species is None else species,
            variant=q['variant'] if variant is None else variant,source=rng.randrange(4)))
    desired=[1,-1] if label=='BOTH' else [-1] if label=='DOWN' else [1]
    for desired_sign in desired:
        interior=nodes[cursor:cursor+depth-1];cursor+=depth-1;chain=[root,*interior,target];product=1
        first=len(rows)
        for i,(a,b) in enumerate(zip(chain,chain[1:])):
            sign=rng.choice([-1,1]) if i<depth-1 else desired_sign*product
            edge(a,b,sign);product*=sign
        if label=='UNKNOWN':
            broken=rows[first+rng.randrange(depth)]
            broken[rng.choice(['species','variant'])]^=1
    # Context and direction traps are present in every answer class.
    edge(root,target,rng.choice([-1,1]),species=1-q['species'])
    edge(target,root,rng.choice([-1,1]))
    decoys=nodes[cursor:]
    while len(rows)<record_count:
        a=rng.choice([root,*decoys]);b=rng.choice(decoys)
        if a==b:continue
        edge(a,b,rng.choice([-1,1]),rng.randrange(2),rng.randrange(2))
    assert len(rows)==record_count
    ids=list(range(record_count));rng.shuffle(ids)
    for r,i in zip(rows,ids):r['id']='F'+str(i)
    rng.shuffle(rows)
    case=dict(id=identifier,namespace=identifier,kind='reason',depth=depth,query=q,records=rows)
    assert solution(case)['answer']==label
    return case


def build(path):
    splits={}
    for split,count,depths,nrecords,seed in [('train',384,[1,2],10,11000),('dev',48,[1,2],10,21000),
            ('iid',96,[1,2],10,31000),('structure',96,[3,4],12,41000)]:
        cases=[]
        for i in range(count):
            depth=depths[(i//4)%len(depths)]
            cases.append(world(f'{split}-{i:04d}',seed+i,depth,LABELS[i%4],nrecords))
        splits[split]=cases
    for split,count in [('train',128),('dev',24),('iid',48),('structure',48)]:
        for i,original in enumerate(splits[split][:count]):
            case=copy.deepcopy(original);case['id']+='-atomic';case['kind']='atomic'
            case['record_id']=case['records'][(i*7)%len(case['records'])]['id']
            splits[split].append(case)
    interventions=[]
    for split in ['iid','structure']:
        for case in [c for c in splits[split] if c['kind']=='reason'][:32]:
            original=solution(case)['answer'];changed=None
            for i in range(len(case['records'])):
                for field in ['sign','species','variant']:
                    trial=copy.deepcopy(case);r=trial['records'][i]
                    r[field]=-r[field] if field=='sign' else 1-r[field]
                    if solution(trial)['answer']!=original:
                        trial.update(id=case['id']+'-changed',parent_id=case['id'],changed_record=r['id'],changed_field=field)
                        changed=trial;break
                if changed:break
            assert changed;interventions.append(changed)
    corpus=dict(seed=11000,splits=splits,interventions=interventions,
        semantics='Formal sign algebra on complete bounded worlds; not biological causality',
        field_schema=['record_id','subject','object','sign','species','variant','source'],
        global_entities='(world namespace, local alias); new world namespaces in every split',
        protocol=dict(train_reason=384,train_atomic=128,dev_reason=48,dev_atomic=24,
            test_iid_reason=96,test_structure_reason=96,test_atomic=96,interventions=64,
            training_depths=[1,2],heldout_structure_depths=[3,4],
            fairness='Same training sample sequence and target traces; text has at least as many trainable parameters',
            model='Pinned LongCat 4-bit; frozen prefix features; last four decoder blocks adapted',
            seeds=[1201,1202],updates=512,maximum_updates=1024,gradient_accumulation=2,
            checkpoint_selection='development teacher-forced target NLL; no held-out outcomes used'))
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(corpus,indent=2)+'\n')
    return corpus
