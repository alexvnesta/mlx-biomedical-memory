"""Frozen richer reasoning tasks and an ARAX qualifier audit.

Synthetic rules are stipulated benchmark semantics, never therapeutic claims.
ARAX cases summarize retained annotations; they do not compose causal effects.
"""

from collections import Counter,defaultdict
import copy
import hashlib
import json
from pathlib import Path
import random

from .relational import infer,record_text


RULES=[
    ('join','Motif Cedar','R0','SUPPORTED iff candidate_gene(C,G), gene_pathway(G,P), candidate_phenotype(C,H), and phenotype_pathway(H,P); otherwise UNKNOWN.'),
    ('context','Motif Cedar','R1','Use only records matching every requested context field. Return UP, DOWN, BOTH, or UNKNOWN from their recorded directions.'),
    ('sign','Motif Quartz','R2','Compose directed regulation signs within the requested context. Multiply signs along paths; return UP, DOWN, BOTH, or UNKNOWN.'),
    ('provenance','Motif Quartz','R3','Count distinct primary_source IDs for each sign. Two sources support a sign. Return SUPPORTED, REFUTED, BOTH, or UNKNOWN.'),
    ('conflict','Motif Vela','R4','Match the claim and every context field. POS or UP supports, NEG or DOWN refutes. Return SUPPORTED, REFUTED, BOTH, or UNKNOWN; never resolve a conflict by majority vote.'),
    ('abduction','Motif Vela','R5','Choose which proposed missing edge completes the candidate_gene/gene_pathway/candidate_phenotype/phenotype_pathway join. Return A, B, MULTIPLE, or NONE.'),
]


def signature(entity,relation,context):
    return json.dumps([entity,relation,context],sort_keys=True,separators=(',',':'))


def request(identifier,entity,relation,context,target):
    scope=', '.join(f'{k}={v}' for k,v in sorted(context.items()))
    prompt='Lookup relation='+relation+'; '+scope+'.\nEntity: {}\nValue:'
    return dict(id=identifier,subject=entity,relation=relation,context=context,target=target,prompt=prompt,
        key=signature(entity,relation,context),
        expressions=[prompt,'Frozen reference. '+prompt,'Read the stored record. '+prompt])


def build(output):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    rng=random.Random(503);requests=[]
    entities=['Entity Mira Tovak','Entity Sova Trevik','Entity Navo Kelmik']
    for entity_index,entity in enumerate(entities):
        for relation in ['gene_label','pathway_label']:
            for species in ['human','mouse']:
                for variant in ['lof','gof']:
                    prefix='G' if relation=='gene_label' else 'P'
                    requests.append(request(f'atomic-{len(requests)}',entity,relation,
                        dict(species=species,variant=variant),prefix+str(rng.randrange(6))))
    motifs=[]
    for family,entity,scope,definition in RULES:
        item=request('motif-'+family,entity,'reasoning_rule',dict(scope=scope),definition)
        item['family']=family;motifs.append(item)
    records=[];cases=[]
    def add_case(family,number,suite='synthetic'):
        identifier=f'{suite}-{family}-{number}';prefix='W'+str(len(cases)).zfill(2)
        root=prefix+'C';local=[]
        def edge(subject,relation,obj,sign=None,context=None,source='S0',**extra):
            r=dict(id=f'{prefix}F{len(local)}',subject=subject,relation=relation,object=obj,
                context=context or {},primary_sources=[source],**extra)
            if sign:r['sign']=sign
            local.append(r);return r
        p=dict(root=root);mode=number%4
        if family in ['join','abduction']:
            edge(root,'candidate_gene',prefix+'G');edge(prefix+'G','gene_pathway',prefix+'P')
            edge(root,'candidate_phenotype',prefix+'H')
            missing=dict(id=prefix+'OPTION',subject=prefix+'H',relation='phenotype_pathway',object=prefix+'P',context={},primary_sources=['HYPOTHETICAL'])
            if family=='join':
                edge(prefix+'H','phenotype_pathway',prefix+('P' if mode%2==0 else 'Q'))
                question=f'Does candidate {root} meet the stored joint gene/pathway and phenotype/pathway constraint?'
            else:
                a=copy.deepcopy(missing);b={**missing,'id':prefix+'OPTIONB','object':prefix+'Q'}
                if mode==1:a,b=b,a
                if mode==2:a={**a,'object':prefix+'Z'}
                if mode==3:b={**b,'object':prefix+'P'}
                p['options']={'A':a,'B':b}
                question=(f'Which hypothetical additional edge completes the stored join for {root}? '
                    f"A: {a['subject']} {a['relation']} {a['object']}; B: {b['subject']} {b['relation']} {b['object']}. "
                    'These options are unproven, not existing evidence.')
        elif family=='sign':
            p.update(target=prefix+'T',context=dict(species='human',variant='lof'))
            signs=['DOWN','DOWN','UP'] if mode==0 else ['UP','DOWN','UP']
            edge(root,'regulates',prefix+'G',signs[0],p['context'])
            edge(prefix+'G','regulates',prefix+'P',signs[1],p['context'])
            if mode!=3:edge(prefix+'P','regulates',prefix+'T',signs[2],p['context'])
            if mode==2:
                product=1
                for s in signs:product*=1 if s=='UP' else -1
                edge(root,'regulates',prefix+'T','DOWN' if product==1 else 'UP',p['context'])
            edge(root,'regulates',prefix+'T','UP',dict(species='mouse',variant='lof'))
            question=f'Under the stored formal sign rule, what direction follows from {root} to {p["target"]} in human lof context?'
        elif family=='context':
            p.update(context=dict(species='human',variant='lof'),relation='effect')
            if mode!=3:edge(root,'effect',prefix+'T','UP' if mode!=1 else 'DOWN',p['context'])
            if mode==2:edge(root,'effect',prefix+'T','DOWN',p['context'])
            edge(root,'effect',prefix+'T','DOWN',dict(species='mouse',variant='lof'))
            edge(root,'effect',prefix+'T','UP',dict(species='human',variant='gof'))
            question=f'What recorded directions apply to {root} effect records specifically in human lof context?'
        elif family in ['provenance','conflict']:
            p.update(context=dict(species='human',variant='lof'),relation='claim')
            positive=mode in [0,2];negative=mode in [1,2]
            for sign,include in [('POS',positive),('NEG',negative)]:
                if include:
                    edge(root,'claim',prefix+'T',sign,p['context'],source='S1')
                    edge(root,'claim',prefix+'T',sign,p['context'],source='S2')
            if mode==3 and family=='provenance':
                for aggregator in ['A1','A2','A3']:
                    edge(root,'claim',prefix+'T','POS',p['context'],source='S1',aggregators=[aggregator])
            edge(root,'claim',prefix+'T','NEG',dict(species='mouse',variant='lof'),source='S3')
            question=(f'Assess the claim for {root} in human lof context using the stored '
                +('distinct-primary-source threshold.' if family=='provenance' else 'explicit support/refutation rule.'))
        else:raise ValueError(family)
        core=[r['id'] for r in local]
        # Disconnected same-vocabulary distractors cannot certify the target.
        edge(prefix+'DECOY','claim',prefix+'OTHER','POS',dict(species='human',variant='lof'),source='S9')
        case=dict(id=identifier,suite=suite,family=family,params=p,record_ids=[r['id'] for r in local],
            core_ids=core,question=question,motif_id='motif-'+family)
        answer,proofs=infer(case,local);case.update(answer=answer,proofs=[sorted(s) for s in proofs])
        records.extend(local);cases.append(case)
    for family,*_ in RULES:
        for i in range(8):add_case(family,i)
    # Dev worlds are never included in test denominators or memory fitting.
    dev_start=len(cases)
    for family,*_ in RULES:add_case(family,20,'dev')
    dev_cases=cases[dev_start:];cases=cases[:dev_start]
    arax_receipts=_add_arax(records,cases)
    interventions=[]
    by_id={r['id']:r for r in records}
    for family,*_ in RULES:
        original=next(c for c in cases if c['suite']=='synthetic' and c['family']==family)
        changed=copy.deepcopy(original);changed['id']=original['id']+'-intervention';changed['parent_id']=original['id']
        local=[copy.deepcopy(by_id[i]) for i in original['record_ids']]
        target=next(r for r in local if r['id'] in original['core_ids'])
        if family in ['context','sign','conflict']:target['sign']='DOWN' if target.get('sign') in ['UP','POS'] else 'UP'
        elif family=='provenance':
            for r in local:
                if r.get('sign')=='POS':r['primary_sources']=['S1']
        elif family=='join':next(r for r in local if r['relation']=='gene_pathway')['object']='ChangedPathway'
        else:next(r for r in local if r['relation']=='gene_pathway')['object']='ChangedPathway'
        answer,proofs=infer(changed,local);changed.update(answer=answer,proofs=[sorted(s) for s in proofs])
        interventions.append(dict(case=changed,replacement_records=local))
    corpus=dict(seed=503,atomic_requests=requests,motifs=motifs,records=records,cases=cases,
        dev_cases=dev_cases,interventions=interventions,arax_receipts=arax_receipts,
        protocol=dict(memory_fitting='24 atomic tuple values plus six generic rule texts; no composed answers, entity-specific proofs, or held-out cases fitted',
            rule_routing='Explicit public motif subject and scope; oracle rule addressing is shared across conditions, not a learned semantic router',
            model='Pinned native LongCat 4-bit; same delta-fitting recipe, at most 25 steps',
            graph_budget=dict(rounds=4,lookups=6,per_round=2,records_per_lookup=16),
            rag='SQLite FTS5 BM25 top12 over the global record corpus; rules supplied as text',
            synthetic='Stipulated relational and sign algebra, not biological causal laws',
            arax='Blinded retained metadata audit; no signed-path causal inference on real ARAX edges'))
    (out/'dataset.json').write_text(json.dumps(corpus,indent=2)+'\n')
    print(json.dumps(dict(atomic=len(requests),motifs=len(motifs),records=len(records),test_cases=len(cases),
        dev_cases=len(dev_cases),answers=dict(Counter(c['answer'] for c in cases)))))
    return corpus


def _add_arax(records,cases):
    receipts=[]
    root=Path('data/processed/arax-anchored')
    if not root.exists():return receipts
    for disease_index,path in enumerate(sorted(root.glob('*.snapshot.json'))):
        snapshot=json.loads(path.read_text());groups=defaultdict(list)
        for edge in snapshot['edges']:
            q={v['qualifier_type_id'].split(':')[-1]:v['qualifier_value'] for v in edge.get('qualifiers') or []}
            if q.get('object_direction_qualifier') not in ['increased','decreased']:continue
            if not edge['object'].startswith('NCBIGene:'):continue
            groups[(str(q.get('species_context_qualifier','')),str(q.get('object_aspect_qualifier','')),
                q['object_direction_qualifier'])].append((edge,q))
        picked=[]
        while len(picked)<12 and groups:
            for key in sorted(list(groups)):
                picked.append(groups[key].pop(0))
                if not groups[key]:del groups[key]
                if len(picked)==12:break
        local=[];aliases={};gene=f'ARAXGene{disease_index}'
        for i,(edge,q) in enumerate(picked):
            alias=aliases.setdefault(edge['subject'],f'ARAXEntity{disease_index}_{len(aliases)}')
            context={k:q[n] for k,n in [('species','species_context_qualifier'),('aspect','object_aspect_qualifier'),
                ('form','subject_form_or_variant_qualifier')] if n in q}
            r=dict(id=f'AR{disease_index}F{i}',subject=alias,relation=edge['predicate'].split(':')[-1],object=gene,
                sign='UP' if q['object_direction_qualifier']=='increased' else 'DOWN',context=context,
                primary_sources=[s['resource_id'] for s in edge.get('sources',[]) if s['resource_role']=='primary_knowledge_source'],
                aggregators=[s['resource_id'] for s in edge.get('sources',[]) if s['resource_role']=='aggregator_knowledge_source'])
            local.append(r)
            receipts.append(dict(record_id=r['id'],disease=path.stem,snapshot_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                original_edge_sha256=hashlib.sha256(json.dumps(edge,sort_keys=True).encode()).hexdigest(),
                original_predicate=edge['predicate'],qualified_predicate=q.get('qualified_predicate'),
                warning='Recorded direction of a qualified aspect, not proof of biological causation or treatment benefit'))
        if not local:continue
        contexts=Counter((r['context'].get('species'),r['context'].get('aspect')) for r in local)
        species,aspect=contexts.most_common(1)[0][0]
        context={k:v for k,v in [('species',species),('aspect',aspect)] if v is not None}
        anchor=next(r for r in local if all(r['context'].get(k)==v for k,v in context.items()))
        for j,family in enumerate(['context','context','provenance','conflict']):
            query_context=dict(context)
            if j==1:query_context['species']='NCBITaxon:9606' if species!='NCBITaxon:9606' else 'NCBITaxon:10090'
            params=dict(root=gene,focus='object',context=query_context)
            if family in ['provenance','conflict']:params.update(subject=anchor['subject'],relation=anchor['relation'])
            desc=', '.join(f'{k}={v}' for k,v in query_context.items())
            question=(f'Summarize recorded incoming directions affecting {gene}, restricted to {desc}.' if family=='context' else
                f"Assess retained direction evidence for {anchor['subject']} {anchor['relation']} {gene}, restricted to {desc}, under the stored {family} rule. UP supports and DOWN refutes this stipulated annotation claim.")
            case=dict(id=f'arax-{disease_index}-{j}',suite='arax',family=family,params=params,
                record_ids=[r['id'] for r in local],core_ids=[r['id'] for r in local],question=question,motif_id='motif-'+family)
            answer,proofs=infer(case,local);case.update(answer=answer,proofs=[sorted(s) for s in proofs]);cases.append(case)
        records.extend(local)
    return receipts
