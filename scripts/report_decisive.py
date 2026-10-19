"""Summarize fixed trained readers and expose every unseen-world generation."""
import hashlib
import html
import json
from pathlib import Path
import random
import statistics


def read(path):return json.loads(Path(path).read_text())
def lines(path):return [json.loads(s) for s in Path(path).read_text().splitlines()]
def mean(values):return statistics.mean(values) if values else None


def aggregate(rows):
    result={}
    for stage in ['iid','structure','intervention','binding','order','masked']:
        selected=[r for r in rows if r['stage']==stage]
        for kind in ['reason','atomic']:
            subset=[r for r in selected if r['kind']==kind]
            if subset:result[stage+'_'+kind]=dict(n=len(subset),correct=sum(r['correct'] for r in subset),
                proof=sum(r['proof_valid'] for r in subset),invalid=sum(r['invalid'] for r in subset),
                mean_seconds=mean([r['wall_seconds'] for r in subset]))
    original={r['case_id']:r for r in rows if r['stage'] in ['iid','structure'] and r['kind']=='reason'}
    for stage in ['intervention','binding','order']:
        pairs=[(original[r['parent_id']],r) for r in rows if r['stage']==stage]
        result[stage+'_pairs']=dict(n=len(pairs),both_labels=sum(a['correct'] and b['correct'] for a,b in pairs),
            both_proofs=sum(a['proof_valid'] and b['proof_valid'] for a,b in pairs),
            changed_output=sum((a['parsed'] or {}).get('answer')!=(b['parsed'] or {}).get('answer') for a,b in pairs))
    atoms=[r for r in rows if r['stage'] in ['iid','structure'] and r['kind']=='atomic']
    result['label_breakdown']={stage:{label:dict(n=len(group),proof=sum(r['proof_valid'] for r in group),correct=sum(r['correct'] for r in group)) for label in ['UP','DOWN','BOTH','UNKNOWN'] if (group:=[r for r in rows if r['stage']==stage and r['kind']=='reason' and r['expected']['answer']==label])} for stage in ['iid','structure']}
    result['depth_breakdown']={str(depth):dict(n=len(group),proof=sum(r['proof_valid'] for r in group),correct=sum(r['correct'] for r in group)) for depth in [1,2,3,4] if (group:=[r for r in rows if r['stage'] in ['iid','structure'] and r['kind']=='reason' and r['depth']==depth])}
    result['atomic_fields']={k:dict(n=len(atoms),correct=sum((r['parsed'] or {}).get(k)==r['expected'][k] for r in atoms))
        for k in ['record','subject','object','sign','species','variant','source']}
    masked=[r for r in rows if r['stage']=='masked']
    if masked:result['mask_comparison']=dict(n=len(masked),normal_proofs=sum(original[r['case_id']]['proof_valid'] for r in masked),
        masked_proofs=sum(r['proof_valid'] for r in masked),normal_labels=sum(original[r['case_id']]['correct'] for r in masked),masked_labels=sum(r['correct'] for r in masked))
    return result


def pct(value,field='proof'):return f"{value[field]}/{value['n']} ({100*value[field]/value['n']:.1f}%)"


def main():
    allrows=[];summary={'readers':{},'by_seed':{},'training':{}};gate=read('runs/decisive-development-gate.json');summary['development_gate']=gate
    for mode in ['text','memory']:
        for seed in [1201,1202]:
            training=Path(f'runs/decisive-{mode}-{seed}');evaluation=Path(str(training)+'-test')
            assert read(evaluation/'summary.json')['completed']
            rows=lines(evaluation/'predictions.jsonl');allrows+=rows
            key=f'{mode}-{seed}';summary['by_seed'][key]=aggregate(rows)
            summary['training'][key]=dict(summary=read(training/'summary.json'),selection=read(training/'generated-selection.json'))
        summary['readers'][mode]=aggregate([r for r in allrows if r['mode']==mode])
    assert len(allrows)==1920
    # A paired world bootstrap conditional on these two fixed fitted seeds.
    # Seeds are averaged within a world; they are not counted as independent worlds.
    differences=[]
    for stage in ['iid','structure']:
        ids=sorted({r['case_id'] for r in allrows if r['stage']==stage and r['kind']=='reason'})
        delta=[]
        for identifier in ids:
            scores={mode:mean([int(r['proof_valid']) for r in allrows if r['stage']==stage and r['case_id']==identifier and r['mode']==mode]) for mode in ['text','memory']}
            delta.append(scores['memory']-scores['text'])
        rng=random.Random(1210);samples=sorted(mean(rng.choices(delta,k=len(delta))) for _ in range(5000))
        differences.append(dict(stage=stage,memory_minus_text=mean(delta),world_bootstrap_95=[samples[125],samples[4874]],
            scope='World resampling conditional on two fixed fitted readers; no population-level training-seed inference.'))
    summary['paired_world_proof_differences']=differences
    latency=lines('runs/decisive-latency/predictions.jsonl');summary['latency']={}
    for mode in ['text','memory']:
        for cached in [False,True]:
            rows=[r for r in latency if r['mode']==mode and r['prefix_cached']==cached]
            summary['latency'][f'{mode}-'+('reused' if cached else 'fresh')]=dict(n=len(rows),
                median_seconds=statistics.median(r['wall_seconds'] for r in rows),
                median_ttft=statistics.median(r['time_to_first_token_seconds'] for r in rows),
                median_prompt_tokens=statistics.median(r['prompt_tokens'] for r in rows),
                median_evaluated_tokens=statistics.median(r['evaluated_prompt_tokens'] for r in rows))
    summary['latency_summary']=read('runs/decisive-latency/summary.json')
    summary['graph_reference']=read('runs/decisive-graph-reference/summary.json')
    summary['graph_median_seconds']=statistics.median(r['seconds'] for r in lines('runs/decisive-graph-reference/predictions.jsonl') if r['kind']=='reason')
    Path('runs/decisive-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    table=['| Reader | Fresh-world labels | Fresh-world valid proofs | Depth 3–4 labels | Depth 3–4 valid proofs | Atomic exact |',
        '|---|---:|---:|---:|---:|---:|']
    for mode in ['text','memory']:
        s=summary['readers'][mode];atomic=dict(n=96*2,correct=s['iid_atomic']['correct']+s['structure_atomic']['correct'])
        table.append('| '+mode+' | '+' | '.join([pct(s['iid_reason'],'correct'),pct(s['iid_reason']),pct(s['structure_reason'],'correct'),pct(s['structure_reason']),pct(atomic,'correct')])+' |')
    table.append('| Exact graph traversal | 100% | 100% | 100% | 100% | 100% |')
    print('\n'.join(table))
    Path('runs/decisive-table.md').write_text('\n'.join(table)+'\n')
    render(summary,allrows)


def render(summary,rows):
    payload=json.dumps(dict(summary=summary,rows=rows),ensure_ascii=False).replace('<','\\u003c')
    template=Path('scripts/decisive-report-template.html').read_text()
    Path('runs/decisive.html').write_text(template.replace('__PAYLOAD__',payload))


if __name__=='__main__':main()
