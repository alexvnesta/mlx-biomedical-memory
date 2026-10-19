"""Executable reference semantics for stipulated rules, not biological causality."""

from collections import defaultdict
import json
import re
import sqlite3


def matches_context(record, context):
    return all(record.get('context', {}).get(key) == value for key, value in context.items())


def selected(case, records):
    p=case['params'];focus=p.get('focus','subject')
    return [r for r in records if r[focus]==p['root']
        and (not p.get('subject') or r['subject']==p['subject'])
        and (not p.get('relation') or r['relation']==p['relation'])
        and matches_context(r,p.get('context',{}))]


def four_value(positive, negative, direction=False):
    if positive and negative:return 'BOTH'
    if positive:return 'UP' if direction else 'SUPPORTED'
    if negative:return 'DOWN' if direction else 'REFUTED'
    return 'UNKNOWN'


def joins(root, records):
    index=defaultdict(list)
    for r in records:index[(r['subject'],r['relation'])].append(r)
    proofs=[]
    for a in index[(root,'candidate_gene')]:
        for b in index[(a['object'],'gene_pathway')]:
            for c in index[(root,'candidate_phenotype')]:
                for d in index[(c['object'],'phenotype_pathway')]:
                    if b['object']==d['object']:proofs.append({r['id'] for r in [a,b,c,d]})
    return proofs


def infer(case, records):
    """Return a label and sufficient proof alternatives under a declared rule."""
    family=case['family'];p=case['params']
    if family=='join':
        proofs=joins(p['root'],records)
        return ('SUPPORTED',proofs) if proofs else ('UNKNOWN',[])
    if family=='abduction':
        valid=[];proofs=[];per_option=[]
        for label,extra in p['options'].items():
            ps=joins(p['root'],[*records,extra])
            if ps:
                option_proofs=[proof-{extra['id']} for proof in ps]
                valid.append(label);proofs.extend(option_proofs);per_option.append(option_proofs)
        if len(per_option)==2:proofs=[a|b for a in per_option[0] for b in per_option[1]]
        return (valid[0] if len(valid)==1 else 'MULTIPLE' if valid else 'NONE'),proofs
    if family=='sign':
        outgoing=defaultdict(list)
        for r in records:
            if r['relation']=='regulates' and r.get('sign') in ['UP','DOWN'] and matches_context(r,p['context']):
                outgoing[r['subject']].append(r)
        paths=defaultdict(list)
        def visit(node,sign,visited,proof):
            if node==p['target']:
                paths[sign].append(set(proof));return
            for r in outgoing[node]:
                if r['object'] not in visited:
                    visit(r['object'],sign*(1 if r['sign']=='UP' else -1),visited|{r['object']},proof+[r['id']])
        visit(p['root'],1,{p['root']},[])
        answer=four_value(bool(paths[1]),bool(paths[-1]),True)
        proofs=([a|b for a in paths[1] for b in paths[-1]] if answer=='BOTH' else paths[1]+paths[-1])
        return answer,proofs
    rows=selected(case,records)
    positive=[r for r in rows if r.get('sign') in ['UP','POS']]
    negative=[r for r in rows if r.get('sign') in ['DOWN','NEG']]
    if family=='provenance':
        def independent_proofs(group):
            result=[]
            for i,a in enumerate(group):
                for b in group[i+1:]:
                    if any(x!=y for x in a.get('primary_sources',[]) for y in b.get('primary_sources',[])):
                        result.append({a['id'],b['id']})
            # One retained record can itself list two distinct primary providers.
            result.extend({r['id']} for r in group if len(set(r.get('primary_sources',[])))>=2)
            return result
        pos,neg=independent_proofs(positive),independent_proofs(negative)
        answer=four_value(bool(pos),bool(neg))
        proofs=[a|b for a in pos for b in neg] if answer=='BOTH' else pos+neg
        return answer,proofs
    if family not in ['context','conflict']:raise ValueError(family)
    answer=four_value(bool(positive),bool(negative),family=='context')
    if answer=='BOTH':proofs=[{a['id'],b['id']} for a in positive for b in negative]
    else:proofs=[{r['id']} for r in positive+negative]
    return answer,proofs


def absence_requirements(case,certificates,records):
    """Sufficient inspected records, or None if exhaustive absence is unproven.

    A certificate is an audited complete tool response, never a model assertion.
    Broader queries qualify; extra context restrictions cannot certify absence.
    """
    p=case['params'];context=p.get('context',{})
    def covered(entity,relation=None,direction='out'):
        for c in certificates:
            if not c.get('complete') or c.get('entity')!=entity:continue
            if c.get('direction','both') not in [direction,'both']:continue
            if c.get('relation') not in [None,relation]:continue
            if any(k not in context or context[k]!=v for k,v in (c.get('context') or {}).items()):continue
            return True
        return False
    family=case['family'];required=set()
    if family in ['context','conflict','provenance']:
        direction='in' if p.get('focus')=='object' else 'out'
        if not covered(p['root'],p.get('relation'),direction):return None
        return {r['id'] for r in selected(case,records)}
    if family=='sign':
        frontier=[p['root']];seen=set()
        while frontier:
            node=frontier.pop()
            if node in seen or node==p['target']:continue
            seen.add(node)
            if not covered(node,'regulates'):return None
            for r in records:
                if r['subject']==node and r['relation']=='regulates' and matches_context(r,context):
                    required.add(r['id']);frontier.append(r['object'])
        return required
    if family in ['join','abduction']:
        for relation,next_relation in [('candidate_gene','gene_pathway'),('candidate_phenotype','phenotype_pathway')]:
            if not covered(p['root'],relation):return None
            for r in records:
                if r['subject']==p['root'] and r['relation']==relation:
                    required.add(r['id'])
                    if not covered(r['object'],next_relation):return None
                    required.update(x['id'] for x in records if x['subject']==r['object'] and x['relation']==next_relation)
        return required
    return None


def grade(case, answer, evidence, retrieved_ids,*,certificates=(),records=()):
    correct=answer.strip().upper()==case['answer']
    cited=set(evidence);retrieved=set(retrieved_ids)
    hallucinated=cited-retrieved
    if case['answer'] in ['UNKNOWN','NONE']:
        # An empty selected subset cannot certify absence. Require inspection of
        # the declared complete local core, and citations to that inspected core.
        proof=set(case['core_ids'])<=cited and set(case['core_ids'])<=retrieved
        required=absence_requirements(case,certificates,records) if certificates else None
        if required is not None:proof=required<=cited
    else:
        proof=any(set(option)<=cited for option in case['proofs'])
    return dict(correct=correct,proof_valid=bool(correct and proof and not hallucinated),
        hallucinated_evidence_ids=sorted(hallucinated),
        core_retrieval_coverage=len(set(case['core_ids'])&retrieved)/len(case['core_ids']) if case['core_ids'] else 1.)


def record_text(r):
    fields=[r['id'],f"subject={r['subject']}",f"relation={r['relation']}",f"object={r['object']}"]
    if r.get('sign'):fields.append('direction='+r['sign'])
    fields.extend(f'{key}={value}' for key,value in sorted(r.get('context',{}).items()))
    fields.append('primary_sources='+','.join(r.get('primary_sources',[])))
    if r.get('aggregators'):fields.append('aggregators='+','.join(r['aggregators']))
    return ' | '.join(fields)


class EvidenceGraph:
    """Lookup executes model-chosen parameters; no case IDs or gold paths."""
    def __init__(self, records):
        self.records=records;self.by_id={r['id']:r for r in records}
        self.index=defaultdict(list)
        for r in records:
            self.index[r['subject']].append(r)
            if r['object']!=r['subject']:self.index[r['object']].append(r)
        self.db=sqlite3.connect(':memory:')
        self.db.execute('CREATE VIRTUAL TABLE records USING fts5(text)')
        self.db.executemany('INSERT INTO records(text) VALUES (?)',[(record_text(r),) for r in records])

    def lookup(self, entity, relation=None, context=None, direction='both', limit=16):
        if direction not in ['both','out','in']:raise ValueError('direction must be both/out/in')
        rows=[r for r in self.index.get(entity,[]) if (not relation or r['relation']==relation)
            and (direction=='both' or r['subject' if direction=='out' else 'object']==entity)
            and matches_context(r,context or {})]
        return dict(records=rows[:limit],complete=len(rows)<=limit,total_matching=len(rows))

    def search(self, text, limit=12):
        terms=sorted(set(re.findall(r'\w+',text.casefold())))
        if not terms:return []
        query=' OR '.join('"'+word+'"' for word in terms)
        ids=self.db.execute('SELECT rowid FROM records WHERE records MATCH ? ORDER BY bm25(records) LIMIT ?',
            (query,limit)).fetchall()
        return [self.records[i-1] for (i,) in ids]


def parse_response(text):
    """Accept the first complete JSON action/answer, including fenced output."""
    decoder=json.JSONDecoder()
    for pos,char in enumerate(text):
        if char!='{':continue
        try:obj,_=decoder.raw_decode(text[pos:])
        except json.JSONDecodeError:continue
        if isinstance(obj,dict) and ('answer' in obj or 'lookups' in obj):return obj
    return None
