"""Publish every completed richer-relation outcome and its provenance."""

from collections import defaultdict
import csv
import hashlib
import html
import json
from pathlib import Path
import shutil
import statistics

import numpy as np


FIT=Path('runs/relational-fit-v1');RUN=Path('runs/relational-v1')
TRANSFER=Path('runs/relational-transfer-v1')
BOUNDARY=Path('runs/relational-boundary-v1')
LABELS=dict(gold_text='Gold evidence + text rule',gold_scoped='Gold evidence + scoped rule',
    gold_ngram='Gold evidence + n-gram rule',gold_shuffled='Gold evidence + shuffled rule',
    gold_native='Gold evidence, rule disabled',bm25_text='BM25 RAG + text rule',hybrid_text='Hybrid RAG + text rule',
    agent_text='Graph agent + text rule',agent_scoped='Graph agent + scoped rule',agent_native='Graph agent, rule disabled',
    native='Native model',ngram='Subject n-gram memory',scoped='Scoped tuple memory',shuffled='Shuffled scoped memory',text='Text value supplied')


def read(path):return json.loads(path.read_text())
def lines(path):return [json.loads(line) for line in path.read_text().splitlines()]
def save(path,data):path.write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n')
def cell(rows,field='correct'):
    return f"{sum(r[field] for r in rows)}/{len(rows)} ({100*sum(r[field] for r in rows)/len(rows):.1f}%)" if rows else '—'
def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|',
        *['| '+' | '.join(str(x) for x in row)+' |' for row in rows]])


def main():
    fit=read(FIT/'summary.json');summary=read(RUN/'summary.json')
    assert fit['completed'] and summary['completed']
    assert fit['frozen_constants']['unchanged'] and summary['frozen_constants']['unchanged']
    corpus=read(RUN/'dataset.json');config=read(RUN/'config.json');rows=lines(RUN/'predictions.jsonl')
    manifest=read(FIT/'memory/manifest.json');requests=corpus['atomic_requests']+corpus['motifs']
    shuffled_equal=sum(requests[i]['target']==requests[manifest['permutation'][i+1]-1]['target']
        for i in range(len(corpus['atomic_requests'])))
    assert len(rows)==summary['rows']
    keys={(r['stage'],r['case_id'],r['method'],r.get('form','')) for r in rows}
    assert len(keys)==len(rows),'Duplicate experimental cells'
    expected=set()
    for r in corpus['atomic_requests']+corpus['motifs']:
        for form in (['canonical','question'] if r['id'].startswith('atomic-') else ['canonical']):
            expected.update(('recall',r['id'],m,form) for m in ['native','ngram','scoped','shuffled','text'])
    for c in corpus['cases']:expected.update(('initial',c['id'],m,'') for m in config['methods'])
    for v in corpus['interventions']:
        expected.update(('intervention',v['case']['id'],m,'') for m in ['gold_text','gold_scoped','agent_text','agent_scoped','agent_native'])
    latency=[r for r in rows if r['stage']=='latency']
    assert len(latency)==72 and all(r['generated_tokens']==32 for r in latency)
    expected.update((r['stage'],r['case_id'],r['method'],'') for r in latency)
    assert keys==expected,f'Missing {len(expected-keys)}, unexpected {len(keys-expected)} cells'
    initial=[r for r in rows if r['stage']=='initial'];recall=[r for r in rows if r['stage']=='recall']
    transfer_summary=read(TRANSFER/'summary.json')
    assert transfer_summary['completed'] and transfer_summary['frozen_constants']['unchanged']
    transfer=lines(TRANSFER/'predictions.jsonl');assert len(transfer)==180
    assert all(r['scoped_probe_exact'] and r['scoped_probe_max_error']==0 for r in transfer)
    boundary_summary=read(BOUNDARY/'summary.json')
    assert boundary_summary['completed'] and boundary_summary['frozen_constants']['unchanged']
    boundary=lines(BOUNDARY/'predictions.jsonl');assert len(boundary)==32
    for directory in [FIT,RUN,TRANSFER,BOUNDARY]:
        dest=Path('artifacts')/directory.name;dest.mkdir(exist_ok=True)
        for name in ['config.json','summary.json','dataset.json','training.jsonl','dev-gold.jsonl','predictions.jsonl','embedding-receipt.json','document-embeddings.npz']:
            if (directory/name).exists():shutil.copy2(directory/name,dest/name)
        for name in ['source','memory']:
            if (directory/name).exists():shutil.copytree(directory/name,dest/name,dirs_exist_ok=True)
        if (directory/'run_relational.py').exists():shutil.copy2(directory/'run_relational.py',dest/'run_relational.py')
        if (directory/'targets').exists():
            (dest/'targets').mkdir(exist_ok=True)
            for p in (directory/'targets').glob('*.json'):shutil.copy2(p,dest/'targets'/p.name)
    for directory in sorted(Path('runs').glob('relational-dev-v*')):
        if (directory/'summary.json').exists():
            shutil.copytree(directory,Path('artifacts')/directory.name,dirs_exist_ok=True)
    metrics=[]
    def group(method,suite=None,family=None):
        return [r for r in initial if r['method']==method and (suite is None or r['suite']==suite) and (family is None or r['family']==family)]
    for method in config['methods']:
        for suite in ['synthetic','arax']:
            rs=group(method,suite)
            metrics.append(dict(method=method,suite=suite,n=len(rs),correct=sum(r['correct'] for r in rs),
                proof_valid=sum(r['proof_valid'] for r in rs),median_wall_seconds=statistics.median(r['wall_seconds'] for r in rs)))
    dest=Path('artifacts/relational-v1');save(dest/'metrics.json',metrics)
    arax_audit=read(Path('runs/relational-arax-audit.json'))
    assert arax_audit['dataset_sha256']==config['dataset_sha256']
    assert sum(r['verified_records'] for r in arax_audit['projections'])==36
    shutil.copy2('runs/relational-arax-audit.json',dest/'arax-audit.json')
    shutil.copy2('runs/relational-validation.json',dest/'validation.json')
    audit_source=Path('scripts/audit_relational_arax.py')
    if hashlib.sha256(audit_source.read_bytes()).hexdigest()!=arax_audit['audit_source_sha256']:
        audit_source=dest/'arax-audit.py'
    assert hashlib.sha256(audit_source.read_bytes()).hexdigest()==arax_audit['audit_source_sha256']
    if audit_source!=dest/'arax-audit.py':shutil.copy2(audit_source,dest/'arax-audit.py')
    with (dest/'metrics.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(metrics[0]),lineterminator='\n');writer.writeheader();writer.writerows(metrics)
    recall_rows=[]
    for method in ['native','ngram','scoped','shuffled','text']:
        atom=[r for r in recall if r['method']==method and r['suite']=='atomic']
        motif=[r for r in recall if r['method']==method and r['suite']=='motif']
        recall_rows.append([LABELS[method],cell([r for r in atom if r['form']=='canonical']),
            cell([r for r in atom if r['form']=='question']),f"{np.mean([r['teacher_forced_nll'] for r in atom]):.3f}",
            cell(motif),cell(motif,'target_prefix_correct'),f"{np.mean([r['teacher_forced_nll'] for r in motif]):.3f}"])
    reasoning_rows=[]
    for method in config['methods']:
        reasoning_rows.append([LABELS[method],cell(group(method,'synthetic')),cell(group(method,'synthetic'),'proof_valid'),
            cell(group(method,'arax')),cell(group(method,'arax'),'proof_valid')])
    supported=lambda method:sum(r['proof_valid'] for r in group(method))
    conclusion=(f"With complete gold evidence, scoped rules produced {supported('gold_scoped')}/60 supported answers, "
        f"versus {supported('gold_text')}/60 with text rules and {supported('gold_native')}/60 with rules disabled. "
        f"For model-planned graph queries, scoped rules produced {supported('agent_scoped')}/60, "
        f"versus {supported('agent_text')}/60 with text rules. ")
    conclusion+=('This run does not demonstrate an advantage over supplying the rule as text.'
        if supported('gold_scoped')<=supported('gold_text') and supported('agent_scoped')<=supported('agent_text')
        else 'Any condition-specific improvement remains exploratory and requires broader confirmation.')
    planning_rows=[]
    for method in ['gold_text','agent_text','agent_scoped','agent_native','bm25_text','hybrid_text']:
        rs=group(method);successes=sum(r['proof_valid'] for r in rs);seconds=sum(r['wall_seconds'] for r in rs)
        planning_rows.append([LABELS[method],cell(rs),cell(rs,'proof_valid'),
            f"{statistics.median(r['wall_seconds'] for r in rs):.3f}",f"{statistics.mean(r['model_calls'] for r in rs):.2f}",
            f"{statistics.mean(r['tool_calls'] for r in rs):.2f}",f'{seconds/successes:.2f}' if successes else '∞'])
    paired_rows=[]
    initial_lookup={(r['case_id'],r['method']):r for r in initial}
    for method,baseline in [('gold_scoped','gold_text'),('gold_scoped','gold_native'),
            ('gold_ngram','gold_native'),('agent_scoped','agent_text'),('agent_scoped','agent_native')]:
        rs=group(method)
        pairs=[(r,initial_lookup[(r['case_id'],baseline)]) for r in rs]
        paired_rows.append([LABELS[method],LABELS[baseline],
            sum(a['proof_valid'] and not b['proof_valid'] for a,b in pairs),
            sum(b['proof_valid'] and not a['proof_valid'] for a,b in pairs),
            sum(a['proof_valid'] and b['proof_valid'] for a,b in pairs),
            sum(not a['proof_valid'] and not b['proof_valid'] for a,b in pairs)])
    family_rows=[]
    for family in ['join','context','sign','provenance','conflict','abduction']:
        for method in config['methods']:
            rs=group(method,'synthetic',family)
            family_rows.append([family,LABELS[method],cell(rs),cell(rs,'proof_valid')])
    intervention_rows=[]
    lookup={(r['case_id'],r['method']):r for r in initial}
    parents={v['case']['id']:v['case']['parent_id'] for v in corpus['interventions']}
    for method in ['gold_text','gold_scoped','agent_text','agent_scoped','agent_native']:
        rs=[r for r in rows if r['stage']=='intervention' and r['method']==method]
        both=sum(r['correct'] and lookup[(parents[r['case_id']],method)]['correct'] for r in rs)
        intervention_rows.append([LABELS[method],cell(rs),cell(rs,'proof_valid'),f'{both}/{len(rs)}'])
    latency_rows=[]
    for method in ['gold_text','gold_scoped','gold_ngram']:
        rs=[r for r in latency if r['method']==method]
        latency_rows.append([LABELS[method],len(rs),f"{statistics.median(r['prompt_tokens'] for r in rs):.0f}",
            f"{statistics.median(r['wall_seconds'] for r in rs):.3f}",f"{np.percentile([r['wall_seconds'] for r in rs],95):.3f}"])
    diagnostics=[]
    for method in config['methods']:
        rs=group(method)
        diagnostics.append([LABELS[method],sum(r['correct'] and not r['proof_valid'] for r in rs),
            sum(bool(r['hallucinated_evidence_ids']) for r in rs),sum(r['answer']=='INVALID' for r in rs),
            sum(r['tool_errors'] for r in rs)])
    development_rows=[]
    for directory in sorted(Path('runs').glob('relational-dev-v*')):
        if not (directory/'summary.json').exists():continue
        results=lines(directory/'predictions.jsonl')
        for method in ['gold_text','agent_text']:
            subset=[r for r in results if r['method']==method]
            development_rows.append([directory.name,LABELS[method],cell(subset),cell(subset,'proof_valid')])
    transfer_rows=[]
    for suite in ['atomic','motif']:
        for form in ['fitted_bare','fitted_reference']:
            for method in ['native','ngram','scoped']:
                rs=[r for r in transfer if r['suite']==suite and r['form']==form and r['method']==method]
                transfer_rows.append([suite,form,LABELS[method],cell(rs),f"{np.mean([r['teacher_forced_nll'] for r in rs]):.3f}"])
    boundary_rows=[]
    for form in ['fitted_bare','fitted_reference','instruction','question']:
        for method in ['original','newline']:
            rs=[r for r in boundary if r['form']==form and r['method']==method]
            boundary_rows.append([form,method,cell(rs),cell(rs,'literal_prefix_only')])
    audit=dict(completed_cells=len(rows),unique_cells=len(keys),all_expected_cells=True,
        fit_sha256=config['checkpoint_sha256'],dataset_sha256=config['dataset_sha256'],
        interventions=sum(r['stage']=='intervention' for r in rows),latency_requests=len(latency),
        training_entity_leakage=False,shuffled_atomic_same_target=shuffled_equal,
        post_hoc_transfer_cells=len(transfer),scoped_probe_exact_all=True)
    audit['post_hoc_boundary_cells']=len(boundary)
    targets='\n'.join(r['target'] for r in corpus['atomic_requests']+corpus['motifs'])
    assert not any(r['subject'] in targets or r['object'] in targets for r in corpus['records']),'Test entity appears in fitting target text'
    save(dest/'audit.json',audit)
    panels=[('Scoped recall',['Condition','Atomic canonical (24)','Atomic alternate (24)','Atomic mean NLL','Rule exact (6)','Rule target prefix (6)','Rule mean NLL'],recall_rows),
        ('Reasoning and proof support',['Condition','Synthetic answer (48)','Synthetic proof (48)','ARAX answer (12)','ARAX proof (12)'],reasoning_rows),
        ('Retrieval and planning',['Condition','Answer (60)','Proof (60)','Median wall, s','Mean model calls','Mean lookups','Observed s per valid proof'],planning_rows),
        ('Paired proof comparison',['Candidate','Reference','Candidate only','Reference only','Both valid','Neither valid'],paired_rows),
        ('Counterfactual evidence',['Condition','Updated answer (6)','Updated proof (6)','Both paired answers correct'],intervention_rows),
        ('Fixed 32-token latency',['Condition','Requests','Prompt tokens','p50 seconds','p95 seconds'],latency_rows),
        ('Failure audit',['Condition','Correct label, insufficient proof','Unretrieved citations','Missing final answer','Tool errors'],diagnostics),
        ('Synthetic families',['Family','Condition','Answer (8)','Proof (8)'],family_rows),
        ('Post-hoc fitted-context diagnostic',['Suite','Format','Condition','Literal recall','Mean NLL'],transfer_rows),
        ('Post-hoc value-boundary control',['Format','Fitting target','Exact value (4)','Literal prefix only (4)'],boundary_rows),
        ('Development interface checks',['Attempt','Condition','Answer (6)','Proof (6)'],development_rows)]
    text=['# Richer relationships: measured results','',
        'An exploratory MLX comparison of scoped factual addresses and reusable latent rule definitions. '
        'These custom cases do not constitute rare-disease discovery or clinical validation. '
        'The earlier [comparison](COMPARISON.md) remains unchanged. [Frozen protocol](RELATIONAL-PROTOCOL.md).','',
        '**Measured conclusion.** '+conclusion,'',
        'The family table matters: scoped memory has some gains over disabled rules, but these do not establish reliable general rule application. '
        'In particular, the stipulated signed-path cases expose failures even with complete evidence and text rules. '
        'A faster completion at lower answer quality cannot establish practical efficiency. '
        'See [rationale and decision criteria](RATIONALE.md) for the remaining utility, confidence and scaling questions.','',
        f"Fitted 24 scoped values and six generic rules in {fit['fit_seconds']:.1f} seconds. "
        f"Peak Metal allocation: fitting {fit['peak_metal_gb']:.2f} GB; evaluation {summary['peak_metal_gb']:.2f} GB. "
        'All base-model parameter digests match before and after fitting and inference.','',
        f"The scoped checkpoint stores 30 non-null rows in {(FIT/'memory/scoped.safetensors').stat().st_size:,} bytes; "
        f"the jointly allocated n-gram checkpoint stores {fit['allocation']['keys']} non-null keys in "
        f"{(FIT/'memory/ngram/updates.safetensors').stat().st_size:,} bytes. "
        'Values are rounded to BF16 and stored in FP32 containers; small checkpoint size excludes base weights and the external evidence index.','',
        'The scoped interface explicitly routes `(entity, relation, context)` to a residual at the entity token. '
        'It is an engineered adapter, not native n-gram addressing or a faithful EngramEdit reproduction. '
        'The same target fits supply both memory conditions, but scoped rows avoid joint allocation/regularization and use a different number of parameters. '
        'This is not a parameter-matched test isolating addressing alone.','']
    for title,headers,body in panels:
        text.extend(['## '+title,'',table(headers,body),''])
        if title=='Scoped recall':text.extend(['Atomic queries repeat fitted lookup tuples in two formats; they are an access/interference diagnostic, not unseen factual knowledge. '
            'Rule-prefix scoring requires the complete stored definition at the start of a completion. NLL is teacher-forced; none of these are reasoning scores. '
            f'The deranged atomic control has {shuffled_equal}/24 coincidentally identical target values because assigned opaque values can repeat.',''])
        if title=='Reasoning and proof support':text.extend(['Gold evidence supplies the complete local view, not a model-discovered path. '
            'All conditions receive the same explicit rule address. Only rule text versus latent storage differs in the gold comparisons. '
            'Synthetic entities and entity-specific proofs never occur as fitting targets. Each family has eight correlated template instances. '
            'ARAX uses 12 queries over three bounded 12-edge views, retaining annotation scope and direction; it does not infer biology by multiplying graph signs. '
            'All 36 blinded ARAX records were separately checked against retained original edge hashes, including subject/object alias bijections, direction, qualifiers and provider roles.',''])
        if title=='Retrieval and planning':text.extend(['Graph agents choose their own entities, relations, directions and context filters. No gold start entity or hop count is passed separately. '
            'The shared budget is four model calls and six lookups, at most two per call and 16 records per lookup. '
            'BM25 and pinned BGE hybrid retrieve top-12 records globally; online CPU query encoding is included. '
            'Observed seconds per valid proof includes time spent on failed requests and is a cohort efficiency measure, not a clinical time-to-answer guarantee. '
            'Model loading, fitting and document embedding preparation are excluded. Variable output lengths affect ordinary wall times.',''])
        if title=='Paired proof comparison':text.extend(['These are matched outcomes on the same 60 questions, including the 12 ARAX audits. '
            'They describe where one condition succeeds while another fails; no significance or population-level effect is inferred from correlated templates.',''])
        if title=='Counterfactual evidence':text.extend(['Six paired interventions change supplied records with rules and memory fixed. '
            'This measures responsiveness to changed evidence, not automatic learning or neural editing. Paired cases are not additional independent worlds.',''])
        if title=='Fixed 32-token latency':text.extend(['Greedy completions are forced to exactly 32 tokens by suppressing EOS. There are 12 queries, two repetitions and randomized condition order. '
            'Model/OS caches are warm, but no prefix cache is reused. Prompt construction, addressing and tokenization are included; output quality is not matched.',''])
        if title=='Development interface checks':text.extend(['The same six separate development worlds were reused for interface diagnosis only. '
            'Attempts 1 and 2 used raw continuation prompts; attempt 3 uses the pinned native chat template with thinking disabled. '
            'All held-out reasoning and latency conditions use the final chat format. These development counts are not pooled with test outcomes.',''])
        if title=='Post-hoc fitted-context diagnostic':text.extend(['This additional diagnostic was designed after observing weak atomic recall under the benchmark instruction prefix. '
            'It replays both exact fitting prefix contexts, uses the same fixed checkpoint, and does not replace any primary outcome. '
            'For every replay, the loaded scoped bank produces bit-identical full-sequence logits to the fitting-time probe with the same BF16-rounded delta. '
            'This checks injection correctness and distinguishes fitted-context access from transfer; it is not an additional reasoning benchmark.',''])
        if title=='Post-hoc value-boundary control':text.extend(['The first four atomic records, selected by frozen dataset order rather than failure status, were separately refitted with a newline after each target. '
            'The optimizer, 25-step budget, templates and two training prefix contexts are unchanged. Original checkpoints and all primary outcomes remain intact. '
            'The original objective supervises target tokens without a termination token: a low loss for `G1` can coexist with generating `G14D`. '
            'Literal-prefix scoring deliberately ignores the following character and is not factual recall. '
            'This four-record control is exploratory, uses the same underlying records repeatedly, and does not establish reasoning or general update quality.',''])
    text.extend(['## Interpretation limits','',
        'A valid proof means sufficient retrieved record IDs under stipulated executable rules, not a validated biological mechanism. '
        'Unknown/none can be certified by complete relevant lookups or a complete inspected core; an arbitrary empty subset is insufficient. '
        'Complete certificates are generated by the graph backend, never accepted as unsupported model claims. '
        'A model can have a correct label and an invalid evidence packet; those are counted separately.','',
        'The latent bank stores generic rule texts, not an autonomously learned theory of biology. '
        'Good recall need not transfer to rule application. Explicit addressing avoids one interference source but does not learn semantic retrieval or guarantee calibrated confidence. '
        'This is one frozen 4-bit backbone under greedy, thinking-disabled generation with a 192-token call budget; it does not establish the limits of other models, reasoning modes or all latent-memory methods. '
        'Different primary-provider IDs do not prove independent experiments. Tissue-specific reasoning, clinical evidence adjudication, prospective discovery and large-scale editing are not established. '
        'OpenEvidence and web search are not scored on private synthetic rules. '
        'The host is a 64 GiB M5 Max Mac; reported Metal peaks are not total system memory. '
        'The busy host had approximately 12 GiB swap in use around evaluation, so these are local observations rather than isolated hardware benchmarks.','',
        '## Reproduce','', '```bash','uv sync --extra dev --extra longcat',
        'uv run --extra longcat python scripts/prepare_longcat.py --download-model',
        'uv run --extra longcat python scripts/prepare_relational.py --replay --cache-encoder',
        'uv run --extra longcat python scripts/fit_relational_memory.py',
        'uv run --extra longcat python scripts/run_relational.py --dev-only --output runs/relational-dev-v3',
        'uv run --extra longcat python scripts/run_relational.py',
        'uv run --extra longcat python scripts/diagnose_scoped_transfer.py',
        'uv run --extra longcat python scripts/diagnose_value_boundary.py',
        '# Reuse the published audit. Omit --replay to recheck retained raw snapshots:',
        'uv run python scripts/audit_relational_arax.py --replay',
        'uv run python scripts/report_relational.py','uv run python scripts/verify_artifacts.py','```','',
        'The commands assume a fresh checkout with no local run outputs. Preserve completed runs when repeating; diagnostic/report defaults use the published version-1 paths. '
        'The current runner reproduces the final chat interface; earlier development interfaces are retained in their source snapshots. '
        'Inference resume is restricted to identical source, data and checkpoint hashes. '
        'Raw prompts, generated actions, audited tool parameters, citations, target-fit histories, checkpoints and source snapshots are public. '
        'Raw ARAX composite graphs and original entity bindings remain local.','',
        'Sources: [EngramEdit](https://arxiv.org/abs/2610.10533), [ProofWriter](https://arxiv.org/abs/2012.13048), '
        '[Graph-CoT](https://arxiv.org/abs/2404.07103). This is not a reproduction of their published benchmark scores.',''])
    Path('docs/RELATIONAL-RESULTS.md').write_text('\n'.join(text))
    body=[]
    panel_notes={
        'Scoped recall':'Literal value recall differs from target-token loss. See the separately labeled termination control below.',
        'Reasoning and proof support':'Gold evidence is a complete local view. Synthetic rules are stipulated; ARAX scores audit bounded annotations. Proof includes a correct answer plus sufficient retrieved record IDs.',
        'Counterfactual evidence':'Evidence changes while rule vectors remain fixed. This is not autonomous learning.',
        'Fixed 32-token latency':'Warm model, no prefix-cache reuse, forced 32-token output. Quality is not matched. Local busy-host observations, not isolated hardware benchmarks.',
        'Post-hoc fitted-context diagnostic':'Additional diagnostic after the primary run. Loaded scoped residuals match fitting-probe logits exactly; none of these are reasoning scores.',
        'Post-hoc value-boundary control':'First four records, refitted with a newline after the value. Literal-prefix success is not exact recall. This control does not replace primary results.',
        'Development interface checks':'Separate worlds reused only for interface diagnosis. All three attempts are retained, outside the test denominators.'}
    for title,headers,values in panels:
        table_class=' class="reasoning"' if title=='Reasoning and proof support' else ''
        note='<p class="note">'+html.escape(panel_notes[title])+'</p>' if title in panel_notes else ''
        body.append('<section id="'+title.lower().replace(' ','-')+'"><h2>'+title+'</h2>'+note+'<div class="scroll"><table'+table_class+'><thead><tr>'+''.join('<th>'+html.escape(h)+'</th>' for h in headers)+
            '</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row)+'</tr>' for row in values)+'</tbody></table></div></section>')
    compact=[]
    for row in initial:
        item={k:v for k,v in row.items() if k!='calls'}
        item['calls']=[{k:v for k,v in call.items() if k!='prompt'} for call in row['calls']];compact.append(item)
    payload=json.dumps(dict(rows=compact,cases={c['id']:c for c in corpus['cases']},labels=LABELS,
        records={r['id']:r for r in corpus['records']}),ensure_ascii=False).replace('<','\\u003c')
    page=r'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Richer relationships · MLX memory comparison</title>
<style>*{box-sizing:border-box}body{margin:0;background:#f1f5f6;color:#19313f;font:15px/1.55 system-ui}main{max-width:1500px;margin:auto;padding:36px 24px}h1{font-size:38px;line-height:1.15;letter-spacing:-1px}h2{font-size:22px}.kicker{color:#006c7b;text-transform:uppercase;letter-spacing:2px;font-size:12px;font-weight:700}.lede{font-size:19px;max-width:1000px}section{background:#fff;border:1px solid #d9e5e9;border-radius:12px;padding:24px;margin:20px 0;scroll-margin-top:18px}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:11px 12px;border-bottom:1px solid #e0e8ec;text-align:left;white-space:nowrap}th{background:#edf5f6;cursor:pointer}td:first-child{font-weight:650}.reasoning{table-layout:fixed}.reasoning td,.reasoning th{white-space:normal;overflow-wrap:anywhere;padding:10px 8px}.reasoning td:first-child,.reasoning th:first-child{width:32%}tr:hover td{background:#f6fafb}a{color:#006b83}select{font:inherit;max-width:100%;padding:8px;margin:0 8px 12px 0;border:1px solid #b8cbd1;border-radius:6px;background:#fff}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f2f6f8;border-radius:8px;padding:18px;font:13px/1.6 ui-monospace,monospace}.note{color:#536e7b;max-width:1050px}.diagram{display:grid;grid-template-columns:1fr 1fr;gap:16px}.diagram div{background:#eff6f7;border-radius:8px;padding:18px}@media(max-width:750px){main{padding:24px 12px}h1{font-size:30px}section{padding:18px}.diagram{grid-template-columns:1fr}}</style>
<main><div class="kicker">Controlled local experiment · 4-bit LongCat · frozen backbone</div><h1>Can scoped memory handle richer relationships?</h1><p class="lede">24 qualified atomic values, six reusable rules, 48 new synthetic worlds and 12 ARAX annotation audits. Answer correctness and executable evidence support are scored separately.</p>
<p class="lede"><b>Measured conclusion:</b> CONCLUSION</p><p class="note">972 primary outcomes plus 212 separately labeled post-hoc diagnostic outcomes. Explicit rule routing is supplied to every condition. Graph agents select their own queries. Synthetic causal rules are assumptions; ARAX cases summarize retained annotations. This is not a rare-disease discovery or clinical validation study.</p>
<p><a href="#reasoning-and-proof-support">Comparison table</a> · <a href="#fixed-32-token-latency">Timing</a> · <a href="#inspect">Inspect raw outcomes</a></p>
<section><h2>What changed</h2><div class="diagram"><div><b>Previous address</b><p>Entity token suffix → shared n-gram vectors<br>Different contexts can collide.</p></div><div><b>Scoped address</b><p>Entity + relation + context → independent residual<br>Same frozen model; explicit routing.</p></div></div></section>
PANELS
<section id="inspect"><h2>Inspect a reasoning result</h2><select id="method" aria-label="Method"><option value="">All conditions</option></select><select id="family" aria-label="Family"><option value="">All families</option></select><select id="outcome" aria-label="Outcome"><option value="">Any result</option><option value="valid">Valid answer and proof</option><option value="label">Correct label, insufficient proof</option><option value="wrong">Wrong answer</option></select><p id="count"></p><select id="example" aria-label="Example" style="width:100%"></select><div id="detail"></div></section>
<section><h2>Read the boundaries</h2><p>Gold evidence is an oracle control. Correct final labels alone do not establish valid reasoning. Proof checks use declared rules and retrieved records; complete backend lookups can certify absence. Different primary providers do not establish independent experiments. A trained rule definition is not an autonomous scientific theory.</p><p><a href="https://github.com/alexvnesta/mlx-biomedical-memory/blob/codex/mlx-biomedical-memory/docs/RELATIONAL-RESULTS.md">Full public report and protocol</a> · <a href="https://github.com/alexvnesta/mlx-biomedical-memory/tree/codex/mlx-biomedical-memory/artifacts/relational-v1">Raw artifacts</a></p></section></main>
<script>const D=PAYLOAD;const el=x=>document.getElementById(x);let selected=[];for(const m of [...new Set(D.rows.map(r=>r.method))])el('method').add(new Option(D.labels[m],m));for(const f of [...new Set(D.rows.map(r=>r.family))])el('family').add(new Option(f,f));function filter(){const m=el('method').value,f=el('family').value,o=el('outcome').value;selected=D.rows.filter(r=>(!m||m===r.method)&&(!f||f===r.family)&&(!o||(o==='valid'?r.proof_valid:o==='label'?r.correct&&!r.proof_valid:!r.correct)));el('count').textContent=selected.length+' matching outcomes';el('example').replaceChildren();selected.forEach((r,i)=>el('example').add(new Option((r.proof_valid?'✓ ':r.correct?'◐ ':'✗ ')+D.labels[r.method]+' · '+D.cases[r.case_id].question,i)));show()}function show(){const r=selected[Number(el('example').value)],d=el('detail');d.replaceChildren();if(!r)return;for(const [label,value] of [['Question',D.cases[r.case_id].question],['Expected / generated',r.expected+' / '+r.answer],['Model responses',r.calls.map(c=>c.generation).join('\n\n')],['Cited records',JSON.stringify(r.evidence.map(id=>D.records[id]||{id,error:'not a known record'}),null,2)],['Audit',JSON.stringify({proof_valid:r.proof_valid,seconds:r.wall_seconds,model_calls:r.model_calls,tool_calls:r.tool_calls,certificates:r.completeness_certificates},null,2)]]){let h=document.createElement('b'),p=document.createElement('pre');h.textContent=label;p.textContent=value;d.append(h,p)}}for(const id of ['method','family','outcome'])el(id).addEventListener('change',filter);el('example').addEventListener('change',show);filter();document.querySelectorAll('th').forEach(th=>th.addEventListener('click',()=>{const i=th.cellIndex,b=th.closest('table').tBodies[0],asc=th.dataset.asc!=='1';th.dataset.asc=asc?'1':'0';[...b.rows].sort((a,z)=>{let x=a.cells[i].textContent,y=z.cells[i].textContent,nx=parseFloat(x),ny=parseFloat(y);return (asc?1:-1)*(isNaN(nx)||isNaN(ny)?x.localeCompare(y):nx-ny)}).forEach(r=>b.append(r))}));</script></html>'''
    page=page.replace('PANELS',''.join(body)).replace('CONCLUSION',html.escape(conclusion)).replace('PAYLOAD',payload)
    Path('runs/relational.html').write_text(page);Path('artifacts/relational.html').write_text(page)
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('artifacts').rglob('*') if p.is_file() and p.name!='SHA256SUMS.json'}
    save(Path('artifacts/SHA256SUMS.json'),hashes)
    print(table(['Condition','Synthetic answer (48)','Synthetic proof (48)','ARAX answer (12)','ARAX proof (12)'],reasoning_rows));print(json.dumps(audit))


if __name__=='__main__':main()
