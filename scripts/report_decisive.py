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
    for stage in ['iid','structure','depth_only','intervention','binding','order','masked']:
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
    result['label_breakdown']={stage:{label:dict(n=len(group),proof=sum(r['proof_valid'] for r in group),correct=sum(r['correct'] for r in group)) for label in ['UP','DOWN','BOTH','UNKNOWN'] if (group:=[r for r in rows if r['stage']==stage and r['kind']=='reason' and r['expected']['answer']==label])} for stage in ['iid','structure','depth_only']}
    result['depth_breakdown']={str(depth):dict(n=len(group),proof=sum(r['proof_valid'] for r in group),correct=sum(r['correct'] for r in group)) for depth in [1,2,3,4] if (group:=[r for r in rows if r['stage'] in ['iid','depth_only'] and r['kind']=='reason' and r['depth']==depth])}
    result['depth_path_breakdown']={str(depth):dict(n=len(group),proof=sum(r['proof_valid'] for r in group)) for depth in [1,2,3,4] if (group:=[r for r in rows if r['stage'] in ['iid','depth_only'] and r['kind']=='reason' and r['depth']==depth and r['expected']['answer']!='UNKNOWN'])}
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
            rows=lines(evaluation/'predictions.jsonl')
            supplemental=Path(str(training)+'-depth-test')
            assert read(supplemental/'summary.json')['completed']
            rows+=lines(supplemental/'predictions.jsonl');allrows+=rows
            key=f'{mode}-{seed}';summary['by_seed'][key]=aggregate(rows)
            summary['training'][key]=dict(summary=read(training/'summary.json'),selection=read(training/'generated-selection.json'),
                phases=[read(p) for p in sorted(training.glob('summary-*.json'))],
                training_seconds_total=sum(read(p)['seconds'] for p in training.glob('summary-*.json')))
        summary['readers'][mode]=aggregate([r for r in allrows if r['mode']==mode])
    summary['trained_selections']={};summary['training_audit']={}
    for seed in [1201,1202]:
        training=Path(f'runs/decisive-memory-{seed}')
        fitted=[]
        for suffix in ['-trained-test','-trained-depth-test']:
            directory=Path(str(training)+suffix);assert read(directory/'summary.json')['completed']
            fitted+=lines(directory/'predictions.jsonl')
        audit=lines(Path(str(training)+'-training-audit/predictions.jsonl'))
        assert len(audit)==48
        audit_summary={condition:dict(n=len(group),correct=sum(r['correct'] for r in group)) for condition in ['normal','masked','changed'] if (group:=[r for r in audit if r['condition']==condition])}
        originals={r['parent_id']:r for r in audit if r['condition']=='normal'}
        audit_summary['same_response_after_edit']=sum(r['parsed'] is not None and r['parsed']==originals[r['parent_id']]['parsed'] for r in audit if r['condition']=='changed')
        summary['training_audit'][str(seed)]=audit_summary;fitted+=audit
        for row in fitted:row['mode']='memory_trained'
        allrows+=fitted;summary['by_seed'][f'memory_trained-{seed}']=aggregate(fitted)
        summary['trained_selections'][str(seed)]=read(training/'trained-only-selection.json')
    summary['readers']['memory_trained']=aggregate([r for r in allrows if r['mode']=='memory_trained'])
    assert len(allrows)==3616
    # A paired world bootstrap conditional on these two fixed fitted seeds.
    # Seeds are averaged within a world; they are not counted as independent worlds.
    differences=[]
    for stage in ['iid','structure','depth_only']:
        ids=sorted({r['case_id'] for r in allrows if r['stage']==stage and r['kind']=='reason'})
        delta=[]
        for identifier in ids:
            scores={mode:mean([int(r['proof_valid']) for r in allrows if r['stage']==stage and r['case_id']==identifier and r['mode']==mode]) for mode in ['text','memory_trained']}
            delta.append(scores['memory_trained']-scores['text'])
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
    for mode in ['text','memory_trained','memory']:
        s=summary['readers'][mode];atomic=dict(n=96*2,correct=s['iid_atomic']['correct']+s['structure_atomic']['correct'])
        table.append('| '+{'text':'Text + LoRA','memory_trained':'Trained memory','memory':'Registered memory selection'}[mode]+' | '+' | '.join([pct(s['iid_reason'],'correct'),pct(s['iid_reason']),pct(s['depth_only_reason'],'correct'),pct(s['depth_only_reason']),pct(atomic,'correct')])+' |')
    table.append('| Exact graph solver (oracle) | 100% | 100% | 100% | 100% | 100% |')
    print('\n'.join(table))
    Path('runs/decisive-table.md').write_text('\n'.join(table)+'\n')
    render(summary,allrows)
    markdown_report(summary,table)


def markdown_report(summary,table):
    fitted=summary['readers']['memory_trained'];text=summary['readers']['text']
    access=all(x['selected']['atomic_correct']>=22 for x in summary['trained_selections'].values())
    conclusion=('The trained memory reader passed the development atomic-access threshold. The held-out proof scores below measure whether that access transfers to unseen compositions.' if access else
        'The training recipe did not establish reliable memory access at the registered budget. This is a failed prerequisite for the proposed reasoning advantage. The held-out results below compare the actual trained reader with the strongest text reader; they do not establish that reliable memory is intrinsically unable to support reasoning.')
    content=['# Learned memory versus trained text on unseen relationships','',
        '**Historical failed training recipe.** The subsequent [readout repair](READOUT-REPAIR-RESULTS.md) achieved 256/256 fresh complete records. The results below retain the original failed-access experiment unchanged. See the [new retention-gated protocol](COMPOSITION-PROTOCOL.md) for the reasoning follow-up.','',conclusion,'',
        'Two training seeds, a pinned LongCat model, identical sample sequences and supervised path traces. Text has 5,914,624 trainable parameters; memory has 5,719,809 including its codec. Both receive every atomic fact. Longer paths in the main table retain the training bank size and record-ID vocabulary.','',*table,'',
        'Each neural stratum has 96 worlds evaluated by two fixed fitted readers. Atomic exactness covers 96 readout questions per seed across the 10- and 12-record banks. UNKNOWN uses the correct absence label and citations to every record; UP, DOWN and BOTH require executable paths. The interactive report separates these classes. The exact graph solver defines the synthetic gold labels. Its 100% is a rule-execution oracle, not an independent production-KG or ARAX evaluation; it has no LLM verbalization.','',
        'The registered memory selection may retain an untrained checkpoint. It is shown separately and must not be described as successfully trained. The trained-memory diagnostic requires a positive training step and uses only already-scored development results for selection.','',
        '## Changes to evidence and bindings','',
        '| Reader | Edit pairs: both proofs valid | Permuted record IDs: both proofs valid | Reordered rows: both proofs valid |',
        '|---|---:|---:|---:|']
    # The already planned class breakdown isolates actual generated paths from
    # correct UNKNOWN predictions. Keep the registered all-world table above.
    split=content.index('## Changes to evidence and bindings')
    path_table=['## Path-present cases only','',
        'The registered all-world score above includes UNKNOWN. The preplanned class breakdown below uses UP, DOWN and BOTH, where the reader must generate an actual executable path. It prevents correct absence guesses from being mistaken for path reasoning.','',
        '| Reader | Fresh depth 1–2 paths | New depth 3–4 paths, 10 records |',
        '|---|---:|---:|']
    for mode in ['text','memory_trained','memory']:
        row=summary['readers'][mode];values=[]
        for stage in ['iid','depth_only']:
            groups=[row['label_breakdown'][stage][label] for label in ['UP','DOWN','BOTH']]
            values.append(pct(dict(n=sum(x['n'] for x in groups),proof=sum(x['proof'] for x in groups))))
        path_table.append('| '+mode+' | '+' | '.join(values)+' |')
    path_table+=['| Exact graph solver (oracle) | 72/72 (100%) | 72/72 (100%) |','','### Actual paths by designed depth','','Depth 1 requires no multi-hop composition. Depths 2–4 do. UNKNOWN cases are excluded from this table.','','| Depth | Text + LoRA | Trained memory |','|---|---:|---:|']
    for depth in ['1','2','3','4']:
        path_table.append('| '+depth+' | '+pct(text['depth_path_breakdown'][depth])+' | '+pct(fitted['depth_path_breakdown'][depth])+' |')
    path_table+=['']
    content[split:split]=path_table
    for mode in ['text','memory_trained','memory']:
        row=summary['readers'][mode]
        content.append('| '+mode+' | '+' | '.join(pct(dict(n=row[k]['n'],proof=row[k]['both_proofs'])) for k in ['intervention_pairs','binding_pairs','order_pairs'])+' |')
    mask=fitted['mask_comparison'];content+=['',f"On the same {mask['n']} worlds, trained-memory supported proofs change from {mask['normal_proofs']} with the codec to {mask['masked_proofs']} with it disabled.",'',
        '## Fixed-length timing','',
        'Selected text and trained-memory readers from seed 1201, 32 generated tokens, EOS suppressed. Each condition has 24 requests. Warm requests reuse facts/instruction KV states and tokenize only the unprimed suffix. Setup costs include full prefix serialization, tokenization and prefill. Adapter changes and model loading are excluded. These are not timings to a correct answer. The Mac remained shared with other open sessions; small timing differences should not be treated as isolated-host performance.','',
        '| Condition | Median wall time | Median first token | Prompt tokens evaluated |','|---|---:|---:|---:|']
    for name,row in summary['latency'].items():content.append(f"| {name} | {row['median_seconds']:.3f} s | {row['median_ttft']:.3f} s | {row['median_evaluated_tokens']} / {row['median_prompt_tokens']} |")
    content+=['',f"Exact in-process traversal of these small complete graphs takes a median {summary['graph_median_seconds']*1e6:.1f} microseconds per reasoning world, excluding network retrieval and LLM verbalization. A useful speed advantage requires comparable supported-answer quality.",'',
        '## What the result addresses','',
        'This is a new engineered tuple codec plus LoRA, not an EngramEdit replication. Text facts pass through all 14 pretrained blocks; latent records enter before the last four. The local vocabulary is closed, with 16 node aliases. Data use fresh world namespaces and facts, not new biomedical terminology. The original 12-record structural stratum also introduces two untrained record IDs, so the separately frozen 10-record depth control is essential.','',
        'The training-bank causal audit, seed-level scores, field-level readouts, original stress stratum, every development candidate and every raw generation are available in the interactive report and public artifacts. All model runs are serial. Metal allocations are limited to 27 GB; aggregate host pressure and swap are retained without publishing other sessions or application names. Frozen backbone hashes are checked before and after training and evaluation.','',
        'Replacement of supplied records tests updateability without fitting new weights. It does not implement autonomous evidence ingestion, retention across tasks, calibrated biomedical confidence or large-graph scaling. Neither a positive nor a negative score here establishes rare-disease discovery or a comparison with OpenEvidence or open-web retrieval.','',
        '## Reproduce and inspect','',
        '- [Registered protocol](DECISIVE-PROTOCOL.md) and linked pre-test clarifications.',
        '- [Reproduction instructions](DECISIVE-REPRODUCTION.md).',
        '- [Design and interpretation](DECISIVE-DESIGN.md).',
        '- [Full interactive report](../artifacts/decisive.html); download and open locally or serve the artifacts directory.',
        '- [Machine-readable summary](../artifacts/decisive-summary.json).',
        '- [Checksums and independent replay](../scripts/verify_decisive_artifacts.py).','']
    Path('docs/DECISIVE-RESULTS.md').write_text('\n'.join(content))


def render(summary,rows):
    payload=json.dumps(dict(summary=summary,rows=rows),ensure_ascii=False).replace('<','\\u003c')
    template=Path('scripts/decisive-report-template.html').read_text()
    Path('runs/decisive.html').write_text(template.replace('__PAYLOAD__',payload))


if __name__=='__main__':main()
