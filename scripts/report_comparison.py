"""Export the complete controlled comparison, including negative results."""

from collections import Counter, defaultdict
import csv
import hashlib
import html
import json
from pathlib import Path
import re
import shutil
import statistics

import numpy as np

from engram_poc.longcat_keys import matches_answer, normalize_answer


ROOT = Path('runs/comparison-v1')
DEST = Path('artifacts/comparison-v1')
LABELS = dict(base='Native LongCat', memory='Fresh memory', memory_cached='Fresh memory + prefix cache',
    shuffled='Shuffled memory', oracle='Gold facts + LongCat', kg='KG + LongCat',
    kg_cached='KG + LongCat + prefix cache', bm25='BM25 RAG', dense='BGE dense RAG',
    hybrid='Hybrid RAG', kg_only='KG only (supplied query plan)')
ORDER = ['base','shuffled','memory','memory_cached','bm25','dense','hybrid','kg','kg_cached','oracle','kg_only']


def load_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def save(path, data):
    path.write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n')


def score(rows):
    n=len(rows)
    return dict(n=n,correct=sum(r['correct'] for r in rows),
        accuracy=sum(r['correct'] for r in rows)/n if n else None,
        median_seconds=statistics.median(r['wall_seconds'] if 'wall_seconds' in r else r['seconds'] for r in rows) if n else None)


def cell(value):
    return f"{100*value['accuracy']:.1f}% ({value['correct']}/{value['n']})" if value['n'] else '—'


def table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|',
        *['| '+' | '.join(str(x) for x in row)+' |' for row in rows]])


def main():
    summary=json.loads((ROOT/'summary.json').read_text())
    assert summary['completed'] and summary['frozen_constants']['unchanged']
    config=json.loads((ROOT/'config.json').read_text())
    corpus=json.loads((ROOT/'dataset.json').read_text())
    raw=load_lines(ROOT/'predictions.jsonl')
    assert len(raw)==summary['predictions']
    keys=[(r['phase'],r['question_id'],r['style'],r['method']) for r in raw]
    assert len(keys)==len(set(keys)), 'Duplicate experimental cells'
    expected=set()
    for q in corpus['questions']:
        methods=['base','memory','shuffled'] if q['kind']=='locality' else config['methods']
        for style in (['standard','cot'] if q['kind']=='composition' else ['standard']):
            expected.update(('initial',q['id'],style,m) for m in methods)
        if q['suite']!='locality':expected.add(('initial',q['id'],'symbolic','kg_only'))
        if q['suite']=='arax' and (q['kind']!='composition' or q['id'].endswith('-0')):
            style='cot' if q['kind']=='composition' else 'standard'
            for phase in ['update-1','update-2','update-3']:
                expected.update((phase,q['id'],style,m) for m in ['memory','kg','hybrid'])
            expected.add(('rollback',q['id'],style,'memory'))
    assert set(keys)==expected, f'Missing {len(expected-set(keys))}, unexpected {len(set(keys)-expected)} experimental cells'
    questions={q['id']:q for q in corpus['questions']}
    facts={f['id']:f for f in corpus['facts']}
    processed=json.loads(json.dumps(raw))
    changed=next(f for f in corpus['facts'] if f.get('relation')=='selected_gene')
    gene_facts=[f for f in corpus['facts'] if f.get('relation')=='selected_disease']
    corrections=0
    for row in processed:
        if row['phase'].startswith('update-') and row['kind']=='composition' and changed['id'] in row['needed_ids']:
            cycle=int(row['phase'].split('-')[1]); gene=gene_facts[[1,2,0][cycle-1]]
            corrections+=int(set(row['needed_ids'])!={changed['id'],gene['id']})
            row['needed_ids']=[changed['id'],gene['id']]
            retrieved=set(row['retrieved_ids'])
            row['support_coverage']=len(set(row['needed_ids'])&retrieved)/len(row['needed_ids'])
    initial=[r for r in processed if r['phase']=='initial']
    metrics={}
    csv_rows=[]
    for method in ORDER:
        group=[r for r in initial if r['method']==method]
        metrics[method]={}
        for name,kind,style,suite in [('recall','recall','standard',None),
                ('paraphrase','paraphrase','standard',None),('composition','composition','standard',None),
                ('composition_cot','composition','cot',None),('arax_cot','composition','cot','arax'),
                ('mquake_cot','composition','cot','mquake'),('unknown','unknown','standard',None),
                ('locality','locality','standard',None)]:
            selected=[r for r in group if r['kind']==kind and (r['style']==style or method=='kg_only')
                and (suite is None or r['suite']==suite)]
            metrics[method][name]=score(selected)
            csv_rows.append(dict(method=method,metric=name,**metrics[method][name]))
    format_rows=[];format_metrics={}
    for method in ORDER:
        if method=='kg_only':continue
        rs=[r for r in initial if r['method']==method and r['kind']=='composition' and r['style']=='cot']
        tolerant=[]
        for row in rs:
            matches=re.findall(r'\bAnswer\s*:\s*([^\n]+)',row['generation'],re.IGNORECASE)
            answer=matches[-1].strip() if matches else ''
            tolerant.append({**row,'correct':matches_answer(answer,row['answers'])})
        format_metrics[method]=score(tolerant)
        format_rows.append([LABELS[method],cell(score(rs)),cell(score(tolerant)),
            sum(not r['answer'] for r in rs),sum(r['generated_tokens']==128 for r in rs)])
    activation_rows=[]
    for kind in ['recall','paraphrase','composition']:
        rs=[r for r in initial if r['method']=='memory' and r['kind']==kind
            and r['style']==('cot' if kind=='composition' else 'standard')]
        hit=[r for r in rs if set(r['needed_ids'])<=set(r['all_activated_fact_ids'])]
        activation_rows.append([kind,f'{len(hit)}/{len(rs)}',cell(score(hit))])
    DEST.mkdir(parents=True,exist_ok=True)
    if Path('runs/comparison-machine.json').exists():
        shutil.copy2('runs/comparison-machine.json','artifacts/comparison-machine.json')
    probe=Path('runs/longcat-gradient-probe')
    if (probe/'summary.json').exists():
        probe_dest=Path('artifacts/longcat-gradient-probe');probe_dest.mkdir(exist_ok=True)
        shutil.copy2(probe/'summary.json',probe_dest/'summary.json')
        for source,target in [('runs/longcat-gradient-probe.log','failed-before-router-fix.log'),
                ('runs/longcat-gradient-probe-v2.log','passed-after-router-fix.log')]:
            if Path(source).exists():shutil.copy2(source,probe_dest/target)
        save(probe_dest/'scope.json',dict(scope='Engineering feasibility preflight, not a held-out benchmark',
            initial_failure='MLX gather_axis integer expert selection did not have an input VJP',
            fix='Stop differentiation through discrete expert IDs; retain selected score gradients',
            source_snapshot_not_recorded=True))
    for name in ['config.json','dataset.json','summary.json','predictions.jsonl','training.jsonl','fitting.json']:
        shutil.copy2(ROOT/name,DEST/name)
    # The runner reserialized Unicode for its local copy. Publish the exact
    # hashed preparation bytes for replay, and retain that runtime copy too.
    prepared_dataset=Path('data/processed/comparison-v1/dataset.json')
    assert json.loads(prepared_dataset.read_text())==corpus
    assert hashlib.sha256(prepared_dataset.read_bytes()).hexdigest()==config['dataset_sha256']
    shutil.copy2(ROOT/'dataset.json',DEST/'dataset-runtime.json')
    shutil.copy2(prepared_dataset,DEST/'dataset.json')
    shutil.copytree(ROOT/'source',DEST/'source',dirs_exist_ok=True)
    # Small exact residual checkpoints are benchmark outputs, not base weights.
    shutil.copytree(ROOT/'memory',DEST/'memory',dirs_exist_ok=True)
    for p in (ROOT/'targets').glob('*.json'):
        (DEST/'targets').mkdir(exist_ok=True);shutil.copy2(p,DEST/'targets'/p.name)
    save(DEST/'predictions-derived.json',processed)
    save(DEST/'derived-field-correction.json',dict(rows=corrections,
        fields=['needed_ids','support_coverage'],raw_answers_and_scores_unchanged=True,
        reason='Sequential composition support IDs originally retained the initial gene fact. Corrected from the stipulated phase target. Raw predictions are retained.'))
    save(DEST/'metrics.json',metrics)
    save(DEST/'secondary-format-metrics.json',format_metrics)
    with (DEST/'metrics.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(csv_rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(csv_rows)
    shutil.copy2('data/processed/comparison-v1/embeddings.npz',DEST/'embeddings.npz')
    shutil.copy2('data/processed/comparison-v1/embedding-receipt.json',DEST/'embedding-receipt.json')
    private_provenance=Path('data/processed/comparison-v1/arax-provenance-local.json')
    if private_provenance.exists():
        public_provenance={}
        for identifier,entry in json.loads(private_provenance.read_text()).items():
            edge=entry['source_edge']
            public_provenance[identifier]=dict(snapshot_sha256=entry['snapshot_sha256'],
                source_edge_sha256=hashlib.sha256(json.dumps(edge,sort_keys=True).encode()).hexdigest(),
                predicate=edge.get('predicate'),qualifiers=edge.get('qualifiers',[]),sources=edge.get('sources',[]),
                source_direction='chemical_to_gene' if edge['object'].startswith('NCBIGene:') and not edge['subject'].startswith('MONDO:')
                    else 'disease_to_gene' if edge['subject'].startswith('MONDO:') else 'gene_to_disease',
                benchmark_relation=facts[identifier]['relation'],
                warning='Projection of a retained record, not a causal or treatment assertion; original identifiers remain local')
        save(DEST/'arax-provenance-summary.json',public_provenance)
    if Path('runs/mquake-global-consistency.json').exists():
        shutil.copy2('runs/mquake-global-consistency.json',DEST/'mquake-global-consistency.json')
    quality_headers=['Condition','Recall (27)','Paraphrases (20)','Composition direct (36)',
        'Composition CoT (36)','ARAX view CoT (18)','Absent entity (6)','Locality (8)']
    quality_rows=[]
    for method in ORDER:
        quality_rows.append([LABELS[method],*[cell(metrics[method][k]) for k in
            ['recall','paraphrase','composition','composition_cot','arax_cot','unknown','locality']]])
    latency_root=Path('runs/comparison-latency')
    latency_rows=[]; latency_summary=None
    latency_overview={}
    if (latency_root/'summary.json').exists():
        latency_summary=json.loads((latency_root/'summary.json').read_text())
        assert latency_summary['completed']
        shutil.copytree(latency_root,Path('artifacts/comparison-latency'),dirs_exist_ok=True)
        latencies=load_lines(latency_root/'predictions.jsonl')
        for method in ['memory','kg','hybrid']:
            for cached in [False,True]:
                rs=[r for r in latencies if r['method']==method and r['prefix_cached']==cached]
                if not cached:latency_overview[method]=statistics.median(r['wall_seconds'] for r in rs)
                latency_rows.append([LABELS[method], 'Warm prefix' if cached else 'No prefix reuse',len(rs),
                    f"{statistics.median(r['prompt_tokens'] for r in rs):.0f}",
                    f"{statistics.median(r['evaluated_prompt_tokens'] for r in rs):.0f}",
                    f"{statistics.median(r['wall_seconds'] for r in rs):.3f}",
                    f"{np.percentile([r['wall_seconds'] for r in rs],95):.3f}",
                    f"{1000*statistics.median(r['retrieval_seconds'] for r in rs):.2f}"])
    update_rows=[]
    for phase in ['initial','update-1','update-2','update-3','rollback']:
        for method in ['memory','kg','hybrid']:
            rs=[r for r in processed if r['phase']==phase and r['suite']=='arax' and r['method']==method
                and (r['kind']!='composition' or r['question_id'].endswith('-0'))
                and r['style']==('cot' if r['kind']=='composition' else 'standard')]
            if not rs:continue
            affected=[r for r in rs if changed['id'] in questions[r['question_id']]['needed']]
            untouched=[r for r in rs if changed['id'] not in questions[r['question_id']]['needed']]
            update_rows.append([phase,LABELS[method],cell(score(rs)),cell(score(affected)),cell(score(untouched))])
    costs=[]
    for update in summary['updates']:
        costs.append([update['phase'],f"{update['memory_fit_seconds']:.2f}",f"{update['retrieval_reindex_seconds']:.4f}"])
    diagnostic_rows=[];diagnostic_summary=None
    diagnostic_root=Path('runs/comparison-diagnostics')
    if (diagnostic_root/'summary.json').exists():
        diagnostic_summary=json.loads((diagnostic_root/'summary.json').read_text())
        assert diagnostic_summary['completed']
        shutil.copytree(diagnostic_root,Path('artifacts/comparison-diagnostics'),dirs_exist_ok=True)
        diagnostics=load_lines(diagnostic_root/'predictions.jsonl')
        prefix_corrections=0
        for row in diagnostics:
            value,target=normalize_answer(row['answer']),normalize_answer(row['target'])
            prefix=value.startswith(target) and (len(value)==len(target) or not (value[len(target)].isalnum() or value[len(target)]=='_'))
            prefix_corrections+=int(prefix!=row['target_prefix_correct'])
            row['target_prefix_correct']=prefix
        save(Path('artifacts/comparison-diagnostics/predictions-derived.json'),diagnostics)
        save(Path('artifacts/comparison-diagnostics/prefix-boundary-correction.json'),dict(rows=prefix_corrections,
            reason='The intended complete-label prefix metric must accept punctuation such as a comma after the label, while rejecting longer labels such as Quartzoid for Quartz.',
            raw_predictions_retained=True,strict_and_primary_scores_unchanged=True))
        for phase in ['primary-canonical','relation-stress']:
            for form in ['canonical','question','paraphrase']:
                for method in ['base','memory_isolated','memory','oracle']:
                    rs=[r for r in diagnostics if r['phase']==phase and r['method']==method and r['form']==form]
                    if not rs:continue
                    prefix=sum(r['target_prefix_correct'] for r in rs)
                    label='Memory fitted alone' if method=='memory_isolated' else LABELS[method]
                    diagnostic_rows.append([phase,form,label,cell(score(rs)),f'{prefix}/{len(rs)}'])
    abstention_rows=[]
    abstention_root=Path('runs/comparison-abstention')
    if (abstention_root/'summary.json').exists():
        abstention_summary=json.loads((abstention_root/'summary.json').read_text())
        assert abstention_summary['completed']
        shutil.copytree(abstention_root,Path('artifacts/comparison-abstention'),dirs_exist_ok=True)
        abstention=load_lines(abstention_root/'predictions.jsonl')
        for method in ['base','memory','kg','bm25','dense','hybrid']:
            rs=[r for r in abstention if r['method']==method]
            abstention_rows.append([LABELS[method],cell(metrics[method]['unknown']),cell(score(rs))])
    scaling=[]; scaling_json=None
    if Path('runs/lookup-scaling.json').exists():
        scaling_json=json.loads(Path('runs/lookup-scaling.json').read_text())
        shutil.copy2('runs/lookup-scaling.json','artifacts/lookup-scaling.json')
        shutil.copy2('scripts/benchmark_lookup_scaling.py','artifacts/lookup-scaling-source.py')
        for row in scaling_json['rows']:
            scaling.append([row['records'],*[f"{row['timings'][m]['p50_ms']:.6f}" for m in
                ['exact_4gram_address','kg_2hop','sqlite_fts5_top4','dense_cosine_top4']],
                f"{row['storage']['dense_float32_bytes']/1e6:.2f}"])
    fitting=json.loads((ROOT/'fitting.json').read_text())
    targets=[json.loads(p.read_text()) for p in (ROOT/'targets').glob('*.json')]
    manifest=json.loads((ROOT/'memory/manifest.json').read_text())
    memory_bytes=(ROOT/'memory/updates.safetensors').stat().st_size
    locality_changes=[]
    for q in corpus['questions']:
        if q['kind']!='locality':continue
        by_method={r['method']:r for r in initial if r['question_id']==q['id']}
        if by_method['base']['answer']!=by_method['memory']['answer']:
            locality_changes.append(dict(question=q['text'],base=by_method['base']['answer'],memory=by_method['memory']['answer']))
    paired={}
    for comparison in ['kg','hybrid','shuffled','base']:
        counts=Counter()
        lookup={(r['question_id'],r['style']):r for r in initial if r['method']==comparison}
        for row in initial:
            if row['method']=='memory' and row['kind']=='composition' and row['style']=='cot':
                other=lookup[(row['question_id'],row['style'])]
                counts[str((row['correct'],other['correct']))]+=1
        paired[comparison]=dict(counts)
    initial_memory={(r['question_id'],r['style']):r for r in initial if r['method']=='memory'}
    rollback_rows=[r for r in processed if r['phase']=='rollback']
    rollback_equivalence=dict(n=len(rollback_rows),
        identical_answers=sum(r['answer']==initial_memory[(r['question_id'],r['style'])]['answer'] for r in rollback_rows),
        identical_generations=sum(r['generation']==initial_memory[(r['question_id'],r['style'])]['generation'] for r in rollback_rows))
    save(DEST/'audit.json',dict(completed_cells=len(raw),unique_cells=len(keys),
        diagnostic_corrections=corrections,locality_answer_changes=locality_changes,
        rollback_equivalence=rollback_equivalence,
        composition_cot_paired_correctness=paired,
        residual_keys={k:len(v) for k,v in manifest['keys'].items()},residual_checkpoint_bytes=memory_bytes))
    lines=['# LongCat on a 64 GB Mac: controlled comparison','',
        'This experiment tests whether freshly fitted conditional memory supports unseen compositions of stored facts. '
        'It does not test a new disease mechanism or treatment. All completed outcomes and raw generations are published.','',
        f"**Measured result:** fresh memory recalls {cell(metrics['memory']['recall'])} atomic facts, "
        f"versus {cell(metrics['base']['recall'])} for the native model and {cell(metrics['shuffled']['recall'])} after shuffling residuals. "
        f"On unseen composed questions with CoT it reaches {cell(metrics['memory']['composition_cot'])}, "
        f"versus {cell(metrics['kg']['composition_cot'])} for KG + LongCat. "
        'This implementation shows a content-dependent recall effect, but does not earn an advantage over KG/RAG for reasoning. '
        'It is not evidence against every conditional-memory architecture or the paper\'s original BF16 setup.','',
        f"Frozen 4-bit LongCat-Flash-Lite; {len(corpus['facts'])} atomic facts; six MQuAKE cases (two each at 2, 3 and 4 hops); "
        'three ARAX disease views; 97 unique questions; 120 retrieval distractors. '
        'Only individual atomic answers were fitting targets. Three composed variants per case/path are correlated observations.','',
        '```mermaid','flowchart LR',
        '    F[Atomic facts only] --> T[Fit one target vector per fact]',
        '    T --> U[Joint regularized n-gram updates]',
        '    U --> M[Versioned exact-key residual memory]',
        '    Q[Unseen composed question] --> K[Token-key activation]',
        '    M --> K','    K --> B[Frozen LongCat]','    B --> A[Generated answer]',
        '    F --> I[Same-corpus KG and retrieval indexes]',
        '    Q --> R[Retrieve support records]','    I --> R','    R --> C[Same frozen LongCat]',
        '    C --> X[Compare answer quality and cost]','    A --> X','```','',
        '## Answer quality','',table(quality_headers,quality_rows),'',
        'Percentages use strict normalized answer-label matching, with retained aliases. '
        'CoT permits intermediate text and requires a final `Answer:` label. '
        'KG-only is symbolic traversal with a supplied entity and hop count; its values in the composition columns are the same structured answers, not CoT generations. '
        'The supplied query plan also assists KG + LongCat. Gold facts are an oracle support control. '
        'RAG uses actual SQLite FTS5 BM25, pinned BGE-small-en-v1.5 cosine retrieval, or reciprocal-rank fusion, top four records. '
        'These are bounded baselines, not a survey of the best agentic or multi-stage RAG systems.','',
        '### Secondary format sensitivity','',
        table(['Condition','Primary CoT (line-start Answer)','Secondary CoT (inline Answer allowed)',
            'Missing primary answer marker','Hit 128-token limit'],format_rows),'',
        'This secondary parser was added after inspecting early outputs that contained a correct inline `Answer:` marker. '
        'It takes the last explicit answer marker on any line and keeps the same exact-label/alias matcher. '
        'Primary scores are unchanged; all methods use the identical secondary rule. This measures sensitivity to answer formatting, '
        'not a retuned model. Correct final labels do not independently validate every intermediate reasoning statement.','',
        '## Fixed-length latency','',
        table(['Condition','Prefix state','Requests','Total prompt tokens, median','Tokens evaluated, median',
            'Wall p50, s','Wall p95, s','Retrieval p50, ms'],latency_rows) if latency_rows else 'Pending latency artifact.', '',
        'Every measured completion has exactly 32 tokens, with EOS suppressed. '
        'Methods and cache states are randomized within each query; there are two repetitions of 12 queries. '
        'Model loading and one-time prefix priming are excluded from steady-state rows and retained separately. '
        'The first priming request also warms model kernels, so individual priming costs are not a balanced cold-start comparison. '
        'No-prefix-reuse requests still use a warm loaded model. Online CPU BGE query encoding is included for hybrid RAG. '
        'This controls output length; it does not match answer quality or measure time to a correct clinical answer. '
        'The quality table separately tests cached inference because arithmetic changes can change generated answers.','',
        '## Corrections and rollback','',table(['Phase','Condition','All 27 ARAX-view probes','Changed chemical: 3 probes','Other 24 probes'],update_rows),'',
        'Three stipulated corrections redirect one chemical alias to another existing gene alias and then back. '
        'They change the benchmark view, not actual biomedical evidence. Each phase uses six two-hop CoT probes plus recall, paraphrase and absent-entity probes. '
        'Other queries test collateral effects; rollback restores the exact initial memory snapshot.','',
        f"Rollback reproduced **{rollback_equivalence['identical_generations']}/{rollback_equivalence['n']}** initial ARAX-view generated texts exactly, "
        'including both correct and incorrect answers.','',
        table(['Correction','Memory fit + allocation, s','One BGE document encoding + full small-index rebuild, s'],costs),'',
        f"Initial fitting and allocation: **{summary['fit_seconds']:.1f} s**; peak Metal allocation: **{summary['peak_metal_gb']:.2f} GB**. "
        f"Rollback: **{summary['rollback_seconds']*1000:.3f} ms**. "
        f"Initial residual checkpoint: **{memory_bytes/1e6:.3f} MB**, with "
        f"{sum(len(v) for v in manifest['keys'].values())} distinct 2/3/4-gram keys. "
        'Residual tensors are stored as FP32 containers of BF16-rounded values. '
        'These costs exclude model download, loading and initial corpus preparation. Metal allocation excludes other process/system memory; this Mac has used swap.','',
        f"Median final target-fit NLL was {statistics.median(t['history'][-1]['nll'] for t in targets):.5f}; "
        f"joint allocation MSE was {fitting['allocation']['allocation_mse']:.6f}. "
        'These are training/surrogate quantities, not held-out answer accuracy. Low fit loss does not ensure that a new question '
        'will activate and use the residual correctly.','',
        '## Separate failure diagnostics','',
        table(['Task','Prompt form','Condition','Strict answer label','Starts with target label'],diagnostic_rows)
            if diagnostic_rows else 'Pending separate diagnostic artifact.','',
        'These diagnostics were added after early primary outputs, with the editing recipe unchanged. '
        'They are not independent validation or new primary metrics. Canonical prompts are the atomic fitting clozes; '
        'target-prefix matching tolerates continuation after the target and is reported separately from strict answer matching. '
        'A derived-field correction accepts punctuation after a complete label (for example `Futurama,`) while rejecting a longer label such as `Quartzoid` for `Quartz`; raw fields remain available. '
        'The relation stress task fits four facts: two relation types for each of two entities. '
        'Memory fitted alone contains only the tested fact; joint memory allocates all four facts together. '
        'Shared-key counts and allocation error are retained. This separates some effects of editing interference '
        'from failure to use even an isolated fact, without attributing every error to one cause.','',
        'In this four-fact stress task, strict question accuracy drops from 2/4 for separately fitted single-fact memories '
        'to 0/4 for joint storage, while supplied facts give 4/4. Both relations for each subject share all three edit keys. '
        'This is a small interference example, not an estimate of general multi-relation performance.','',
        '## Lookup scaling, separate from inference','',
        table(['Records','Exact address p50, ms','KG two-hop p50, ms','FTS5 top-4 p50, ms','Exhaustive cosine p50, ms','Dense vectors, MB'],scaling)
            if scaling else 'Pending CPU index artifact.','',
        'Synthetic CPU indexes only. Exact address lookup excludes token scanning, CPU/GPU transfers and residual gathering. '
        'KG gets a normalized entity and a fixed two-hop plan. Dense search excludes query encoding and uses exhaustive cosine, not ANN. '
        'Index build/storage details are retained. This does not demonstrate learned-memory accuracy, editing cost or interference at scale.','',
        f"At the observed {sum(len(v) for v in manifest['keys'].values())/len(corpus['facts']):.2f} distinct keys per fact, "
        f"a naive one-million-fact extrapolation would add about {sum(len(v) for v in manifest['keys'].values())/len(corpus['facts'])*3072*4/1000:.1f} GB "
        'of FP32 residual tensors, excluding Python keys and the base model. This is storage arithmetic, not a measured scaling result; '
        'sharing and compression could change it. The current joint allocator constructs a dense key-by-key matrix and uses a dense solve, '
        'so it is explicitly a small-batch implementation. Large edits need sparse/component-wise allocation, bounded working sets and interference tests. '
        'The CPU exact-key wrapper also synchronizes token IDs to host memory. It is not an optimized production memory kernel.','',
        '## What the experiment can establish','',
        'The practical hypothesis is a versioned local factual cache inside an otherwise frozen model: '
        'update selected facts, use them without supplying their text on every request, and roll back independently. '
        'A successful edit alone establishes recall, not multi-hop reasoning. '
        'The composed questions require outputs never used as fitting targets; shuffled updates test whether the fitted content matters. '
        'The anonymized ARAX view removes useful pretrained entity associations, but measures selected-neighbor composition only.','',
        'In this measured implementation, memory is slower than KG + LongCat in both prefix-cache states, '
        'despite using shorter total prompts. Together with the quality deficit, that provides no operational reason '
        'to replace the graph/RAG path with this residual interface. The closed-world test also does not establish '
        'benefit for evidence synthesis, clinical support or rare-disease hypothesis generation.','',
        'For rare-disease evidence work, the graph remains the source of record for relation direction, provenance, qualifiers, contradictions and dates. '
        'A learned residual is not a citation and does not establish calibrated confidence. '
        'Repeated targeted edits are a form of incremental updating; autonomous continuous learning, source adjudication and safe consolidation are unimplemented. '
        'A useful deployment would need stable quality at larger edit counts, reliable abstention, semantic and relation-specific addressing, auditability, '
        'and a latency or context advantage over strong cached KG/RAG at comparable correctness.','',
        table(['Memory question type','All required fact IDs had an activated edit key','Accuracy among those hits'],activation_rows),'',
        'An activation means a mapped 2/3/4-gram key occurred somewhere in the prompt or generated continuation; '
        'it does not prove that the model used the fact correctly. All atomic recall prompts activated their expected keys, '
        'yet some answers were wrong. Key-hit telemetry therefore cannot substitute for calibrated confidence or source verification.','',
        '### Secondary absent-record audit','',
        table(['Condition','Original aliases containing Unknown','Neutral absent aliases'],abstention_rows),'',
        'The initial six absent-entity aliases contain the literal word `Unknown`, so abstention may exploit that cue. '
        'The secondary audit removes it: six neutral chemical aliases are absent from every fact and distractor, with the same question/inference recipe and initial memory checkpoint. '
        'This audit was added after identifying the lexical cue; it is not a clinical confidence calibration study.','',
        '## Limits and external tools','',
        'This is one quantization-aware adaptation, not an exact replication of '
        '[EngramEdit](https://arxiv.org/abs/2610.10533). It uses fixed atomic expression templates and two contexts, '
        'with frequency calibration from this local corpus instead of Wikipedia. '
        'Base Transformer, native embeddings, routing parameters and output weights have identical SHA-256 digests before and after all edits. '
        'The router wrapper preserves forward values while stopping differentiation through discrete expert IDs. '
        'The target-fit objective, quantization, wording and tiny sample all limit conclusions. No test-result tuning was used.','',
        'Target fitting uses at most 25 Adam steps, learning rate 0.5, a 32× initial-representation norm clamp, '
        'KL coefficient 0.0625 and norm coefficient 0.001; the objective can stop below 0.05 at an even-numbered step. '
        'The allocator uses ridge 0.01, reuse 0.05 and length penalties 8/2/1 for 2/3/4-grams. '
        'The canonical expression, three fixed prefixes and the supplied atomic question use two contexts. '
        'Changing a prefix provides less linguistic diversity than independently generated paraphrases. '
        'Targets are fitted independently on the unedited native model, then allocated jointly; later corrections fit against the current edited model.','',
        'OpenEvidence and web search were not evaluated head to head: no authorized benchmark session/API was available, '
        'and public tools cannot be expected to know our private fictional updates or blinded aliases. '
        'Their scores are **not measured**, not zero. A fair clinical comparison needs shared accessible evidence, '
        'real questions, date-controlled sources, expert grading and a defined provenance/cost rubric. '
        '[OpenEvidence](https://portal.openevidence.com/) and a web-enabled RAG workflow are different evidence-facing systems, '
        'not interchangeable baselines for this closed-world edit test.','',
        'The earlier [30-case released-checkpoint replay](LONGCAT-RESULTS.md) and [Qwen experiment](RESULTS.md) '
        'remain separate. Their datasets, prompts, training and edit counts differ; differences cannot isolate an architecture or precision effect.','',
        '## Reproduce and inspect','',
        '```bash','uv sync --extra dev --extra longcat',
        'uv run --extra longcat python scripts/prepare_comparison.py --replay',
        'uv run --extra longcat python scripts/run_comparison.py',
        'uv run --extra longcat python scripts/benchmark_comparison_latency.py',
        'uv run --extra longcat python scripts/diagnose_comparison_memory.py',
        'uv run --extra longcat python scripts/audit_abstention.py',
        'OPENBLAS_NUM_THREADS=1 uv run python scripts/benchmark_lookup_scaling.py',
        'uv run python scripts/report_comparison.py','```','',
        'The `--replay` preparation verifies and copies the exact public corpus/vectors and caches the pinned CPU encoder. '
        'Rebuilding without that flag requires retained ARAX snapshots and the pinned MQuAKE source. '
        'Base weights remain a separate download using `scripts/prepare_longcat.py --download-model`. '
        'Small fitted residual checkpoints, source snapshots, target-fit histories, raw answers, timing and checksums are published. '
        'Raw composite ARAX graphs and the private identity mapping are not redistributed.','',
        '[Machine-readable metrics](../artifacts/comparison-v1/metrics.csv) · '
        '[Raw predictions](../artifacts/comparison-v1/predictions.jsonl) · '
        '[Audit](../artifacts/comparison-v1/audit.json) · '
        '[Derived-field correction](../artifacts/comparison-v1/derived-field-correction.json)','']
    Path('docs/COMPARISON.md').write_text('\n'.join(lines))
    content=[]
    overview_rows=[]
    for method in ['memory','shuffled','bm25','dense','hybrid','kg','kg_only']:
        short='KG only, supplied plan' if method=='kg_only' else LABELS[method]
        recall=metrics[method]['recall'];compose=metrics[method]['composition_cot']
        timing=f"{latency_overview[method]:.3f} s" if method in latency_overview else '—'
        overview_rows.append([short,f"{recall['correct']}/{recall['n']}",f"{compose['correct']}/{compose['n']}",timing])
    for title,headers,data in [
        ('At a glance',['Condition','Atomic recall','Unseen CoT','32-token p50'],overview_rows),
        ('Answer quality',quality_headers,quality_rows),
        ('Format sensitivity · secondary post-hoc check',['Condition','Primary CoT','Inline marker allowed','Missing primary marker','At token limit'],format_rows),
        ('Fixed 32-token latency',['Condition','Prefix state','Requests','Prompt tokens','Evaluated tokens','p50 seconds','p95 seconds','Retrieval ms'],latency_rows),
        ('Corrections and rollback',['Phase','Condition','All 27 probes','Changed chemical (3)','Other probes (24)'],update_rows),
        ('Update cost',['Phase','Memory fit + allocation, seconds','Encode + index rebuild, seconds'],costs),
        ('Separate failure diagnostics',['Task','Prompt form','Condition','Strict label','Target prefix'],diagnostic_rows),
        ('Key hits are not confidence',['Question type','All required fact IDs activated','Accuracy among hits'],activation_rows),
        ('Secondary abstention check',['Condition','Unknown cue in alias','Neutral absent alias'],abstention_rows),
        ('Synthetic CPU index scaling',['Records','Exact address ms','KG two-hop ms','FTS5 ms','Cosine ms','Dense MB'],scaling)]:
        body=''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row)+'</tr>' for row in data)
        anchor=re.sub(r'[^a-z0-9]+','-',title.casefold()).strip('-')
        content.append('<section id="'+anchor+'"><h2>'+title+'</h2><div class="scroll"><table><thead><tr>'+''.join('<th>'+html.escape(h)+'</th>' for h in headers)+
            '</tr></thead><tbody>'+body+'</tbody></table></div></section>')
    payload=json.dumps(dict(rows=processed,questions=questions,labels=LABELS),ensure_ascii=False).replace('<','\\u003c')
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>LongCat memory · Controlled comparison</title><style>
:root{color-scheme:light;--ink:#172b3a;--muted:#536878;--blue:#005f75;--line:#dce4e8}*{box-sizing:border-box}body{margin:0;background:#f2f5f6;color:var(--ink);font:15px/1.55 system-ui,sans-serif}main{max-width:1500px;margin:auto;padding:42px 28px}h1{font-size:clamp(28px,4vw,46px);letter-spacing:-1.4px;line-height:1.12;margin:8px 0 20px}h2{font-size:22px;margin:0 0 18px}.kicker{color:var(--blue);font-size:12px;font-weight:750;letter-spacing:1.8px;text-transform:uppercase}.lede{font-size:19px;max-width:950px}.note{color:var(--muted);max-width:1050px}.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:28px 0}.stat,section{background:white;border:1px solid var(--line);border-radius:12px}.stat{padding:18px}.stat strong{font-size:27px;display:block}.stat span{color:var(--muted);font-size:13px}section{padding:26px;margin:20px 0}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:12px 10px;text-align:left;border-bottom:1px solid var(--line);white-space:nowrap}th{background:#edf4f6;color:#254758;cursor:pointer}td:first-child{font-weight:650}tr:hover td{background:#f7fbfc}a{color:#006b83}select,input{font:inherit;border:1px solid #b5c8d0;border-radius:6px;padding:8px;background:white;margin:0 8px 12px 0}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f1f5f7;padding:18px;border-radius:8px;font:13px/1.6 ui-monospace,monospace}.good{color:#00724d}.bad{color:#b13636}.diagram{display:grid;grid-template-columns:1fr 1fr;gap:20px}.diagram div{background:#f1f6f7;border-radius:10px;padding:20px}.diagram b{display:block;color:#005f75;margin-bottom:10px}.pill{display:inline-block;padding:4px 10px;background:#e7f0f4;border-radius:99px;font-size:12px}footer{color:var(--muted);margin-top:28px}@media(max-width:750px){main{padding:24px 12px}.stats{grid-template-columns:1fr 1fr}section{padding:18px}.diagram{grid-template-columns:1fr}}
</style><main><div class="kicker">Apple silicon · Frozen experiment · All outcomes retained</div>
<h1>Does editable memory earn its place?</h1>
<p class="lede">A controlled comparison of newly fitted LongCat memory, graph traversal and retrieval. The test asks whether independently stored facts support answers never used as training targets.</p>
<p class="note">64 GB M5 Max · 4-bit LongCat-Flash-Lite · Six MQuAKE cases and three anonymized ARAX disease views. These are small, closed-world record-composition tests. They do not demonstrate new biomedical discoveries.</p>
<section><h2>Measured result</h2><p class="lede">VERDICT</p><p class="note">Memory shows a content-dependent recall effect. This implementation has not demonstrated an advantage over KG/RAG for reasoning. The result applies to this small 4-bit adaptation.</p></section>
<div class="stats"><div class="stat"><strong>27</strong><span>atomic facts fitted</span></div><div class="stat"><strong>97</strong><span>unique test questions</span></div><div class="stat"><strong>36</strong><span>unseen composed questions</span></div><div class="stat"><strong>PEAK</strong><span>GB peak Metal allocation</span></div></div>
<section><h2>Two routes to the same stipulated answer</h2><div class="diagram"><div><b>Conditional memory</b>Atomic facts → fit target vectors → allocate shared n-gram updates<br>Question → exact token-key activation → frozen model → answer<br><span class="pill">Composed answers never fitted</span></div><div><b>Graph and retrieval controls</b>Same facts → adjacency / FTS5 / BGE vector index<br>Question → retrieve records → frozen model → answer<br><span class="pill">KG entity and hop plan supplied</span></div></div></section>
CONTENT
<section><h2>How to read these results</h2><p>Quality cells show percent and exact counts. Three variants of one case are correlated. KG-only uses a supplied structured query plan; it does not generate CoT. RAG retrieves four records from 147 documents. Gold facts provide the exact support.</p><p>Fixed-token timing measures 32 generated tokens with EOS suppressed. Prefix setup is retained separately. It does not measure time to a correct answer. Index scaling excludes neural generation, and address lookup excludes the rest of the memory interface.</p><p>Corrections change one stipulated alias relationship. They do not revise biomedical truth. The graph remains necessary for source provenance, qualifiers, contradictory evidence and dates. A fitted vector supplies neither a citation nor calibrated confidence.</p><p><strong>OpenEvidence and web search: not measured.</strong> A clinical comparison requires shared accessible evidence, real questions and expert grading. Public tools cannot be scored fairly on private fictional edits.</p></section>
<section><h2>Inspect individual answers</h2><input id="search" placeholder="Search questions or generations" aria-label="Search"><select id="method" aria-label="Method"><option value="">All conditions</option></select><select id="phase" aria-label="Phase"><option value="">All phases</option></select><select id="outcome" aria-label="Outcome"><option value="">Any outcome</option><option value="wrong">Incorrect</option><option value="right">Correct</option></select><p id="count"></p><select id="example" style="max-width:100%;width:100%" aria-label="Individual answer"></select><div id="detail"></div></section>
<footer><a href="https://github.com/alexvnesta/mlx-biomedical-memory/blob/codex/mlx-biomedical-memory/docs/COMPARISON.md">Full protocol and interpretation</a> · <a href="https://github.com/alexvnesta/mlx-biomedical-memory/tree/codex/mlx-biomedical-memory/artifacts/comparison-v1">Public raw artifacts</a><p>Exploratory MLX adaptation of EngramEdit. Fixed expression templates and local frequency calibration differ from the paper. No clinical confidence claim.</p></footer></main>
<script>const DATA=PAYLOAD;
const byId=x=>document.getElementById(x);let filtered=[];
for(const [value,label] of Object.entries(DATA.labels)){let o=new Option(label,value);byId('method').add(o)}
for(const phase of [...new Set(DATA.rows.map(r=>r.phase))])byId('phase').add(new Option(phase,phase));
function filter(){const s=byId('search').value.toLowerCase(),m=byId('method').value,p=byId('phase').value,o=byId('outcome').value;filtered=DATA.rows.filter(r=>(!m||r.method===m)&&(!p||r.phase===p)&&(!o||(o==='right')===r.correct)&&(!s||(DATA.questions[r.question_id].text+' '+(r.generation||r.answer)).toLowerCase().includes(s)));byId('count').textContent=filtered.length+' matching answers';byId('example').replaceChildren();filtered.forEach((r,i)=>byId('example').add(new Option((r.correct?'✓ ':'✗ ')+r.phase+' · '+DATA.labels[r.method]+' · '+r.style+' · '+DATA.questions[r.question_id].text,i)));show()}
function show(){const r=filtered[Number(byId('example').value)],d=byId('detail');d.replaceChildren();if(!r)return;for(const [label,value] of [['Question',DATA.questions[r.question_id].text],['Expected',r.answers.join(' / ')],['Generated',r.generation||r.answer],['Diagnostics',JSON.stringify({correct:r.correct,seconds:r.wall_seconds,prompt_tokens:r.prompt_tokens,generated_tokens:r.generated_tokens,retrieved:r.retrieved_ids,needed:r.needed_ids,activated:r.all_activated_fact_ids},null,2)]]){const h=document.createElement('b');h.textContent=label;const pre=document.createElement('pre');pre.textContent=value;d.append(h,pre)}}
for(const id of ['search','method','phase','outcome'])byId(id).addEventListener('input',filter);byId('example').addEventListener('change',show);filter();
document.querySelectorAll('th').forEach(th=>th.addEventListener('click',()=>{const i=th.cellIndex,b=th.closest('table').tBodies[0],asc=th.dataset.asc!=='1';th.dataset.asc=asc?'1':'0';[...b.rows].sort((a,z)=>{let x=a.cells[i].textContent,y=z.cells[i].textContent;let nx=parseFloat(x),ny=parseFloat(y);return (asc?1:-1)*(Number.isNaN(nx)||Number.isNaN(ny)?x.localeCompare(y):nx-ny)}).forEach(r=>b.append(r))}));
</script></html>'''
    verdict=(f"Fresh memory: {metrics['memory']['recall']['correct']}/27 atomic recalls; "
        f"{metrics['memory']['composition_cot']['correct']}/36 unseen CoT answers. "
        f"KG + LongCat: {metrics['kg']['composition_cot']['correct']}/36 unseen CoT answers.")
    page=page.replace('VERDICT',html.escape(verdict)).replace('PEAK',f"{summary['peak_metal_gb']:.2f}").replace('CONTENT',''.join(content)).replace('PAYLOAD',payload)
    page=page.replace('<div class="stats">','<nav><a href="#at-a-glance">Overview</a><a href="#answer-quality">Quality</a><a href="#fixed-32-token-latency">Latency</a><a href="#corrections-and-rollback">Corrections</a><a href="#raw-answers">Raw answers</a></nav><div class="stats">')
    page=page.replace('<section><h2>Inspect individual answers</h2>','<section id="raw-answers"><h2>Inspect individual answers</h2>')
    page=page.replace('</style>','nav{display:flex;gap:18px;flex-wrap:wrap;margin:24px 0}section{scroll-margin-top:18px}td:first-child,th:first-child{position:sticky;left:0;background:white}th:first-child{background:#edf4f6}</style>')
    Path('runs/comparison.html').write_text(page)
    Path('artifacts/comparison.html').write_text(page)
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('artifacts').rglob('*')
        if p.is_file() and p.name!='SHA256SUMS.json'}
    save(Path('artifacts/SHA256SUMS.json'),hashes)
    print(table(quality_headers,quality_rows))
    print(json.dumps(dict(predictions=len(raw),diagnostic_corrections=corrections,artifacts=len(hashes))))


if __name__=='__main__':main()
