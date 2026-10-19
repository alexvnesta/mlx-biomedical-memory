"""Compare free proofs only after access has qualified; preserve every denominator."""
import html
import json
from pathlib import Path
import statistics


SEEDS=[2401,2402]
METHODS=dict(text=('Full text','composition-text'),frozen=('Frozen memory encoder','composition-memory'),
    joint=('Joint memory adaptation','composition-joint'))


def ratio(rows,key='proof_valid'):
    return dict(correct=sum(bool(r[key]) for r in rows),total=len(rows))


def show(value):
    return f"{value['correct']}/{value['total']} ({100*value['correct']/value['total']:.1f}%)" if value['total'] else 'n/a'


def main():
    corpus=json.loads(Path('runs/composition-features-v1/dataset.json').read_text())
    gates={name:json.loads(Path('runs',name,'development-gate.json').read_text())
        for name in ['composition-v1','composition-joint-v1']}
    assert json.loads(Path('runs/composition-joint-v1/summary.json').read_text())['completed']
    selections={key:value for gate in gates.values() for key,value in gate['selection'].items()}
    failures=[key for key,value in selections.items() if value is None]
    rows=[];readers={};training={};resources=[]
    for method,(label,prefix) in METHODS.items():
        for seed in SEEDS:
            root=Path('runs')/f'{prefix}-{seed}';test=Path(str(root)+'-test')
            if selections[f'{prefix.removeprefix("composition-")}-{seed}'] is None:
                assert not test.exists()
                continue
            summary=json.loads((test/'summary.json').read_text());assert summary['completed']
            batch=[dict(r,method=method) for r in [json.loads(x) for x in (test/'predictions.jsonl').read_text().splitlines()]]
            readers[method,seed]=batch;rows+=batch
            training[method,seed]=json.loads((root/'generated-selection.json').read_text())['selected']
            resources += [summary['resource'],json.loads((root/'summary.json').read_text())['resource']]
    metrics={};perseed=[]
    for method,(label,_) in METHODS.items():
        batch=[r for r in rows if r['method']==method]
        atomic=[r for r in batch if r['kind']=='atomic' and r['stage'] in ['iid','structure']]
        reason=[r for r in batch if r['kind']=='reason' and r['stage'] in ['iid','structure']]
        positive=[r for r in reason if r['expected']['answer']!='UNKNOWN']
        value=dict(atomic=ratio(atomic,'correct'),depths={str(d):ratio([r for r in positive if r['depth']==d]) for d in [1,2,3,4]},
            iid_labels=ratio([r for r in reason if r['stage']=='iid'],'correct'),iid_proofs=ratio([r for r in reason if r['stage']=='iid']),
            structure_labels=ratio([r for r in reason if r['stage']=='structure'],'correct'),structure_proofs=ratio([r for r in reason if r['stage']=='structure']),
            controls={stage:ratio([r for r in batch if r['kind']=='reason' and r['stage']==stage]) for stage in ['intervention','binding','order']},
            masked_positive=ratio([r for r in batch if r['stage']=='masked' and r['expected']['answer']!='UNKNOWN']),
            median_prompt_tokens=statistics.median(r['prompt_tokens'] for r in reason) if reason else None)
        eligible=[]
        for seed in SEEDS:
            if (method,seed) not in readers:continue
            part=readers[method,seed]
            a=ratio([r for r in part if r['kind']=='atomic' and r['stage'] in ['iid','structure']],'correct')
            eligible.append(a['correct']/a['total']>=.95)
            perseed.append(dict(method=method,seed=seed,atomic=a,
                iid_positive=ratio([r for r in part if r['stage']=='iid' and r['kind']=='reason' and r['expected']['answer']!='UNKNOWN']),
                structure_positive=ratio([r for r in part if r['stage']=='structure' and r['kind']=='reason' and r['expected']['answer']!='UNKNOWN']),
                selected=training[method,seed]))
        value['retention_qualified']=bool(eligible) and all(eligible);value['eligible_seeds']=[s for s in SEEDS if (method,s) in readers];metrics[method]=value
    support={}
    required={}
    for case in corpus['support_readouts']:required.setdefault(case['parent_id'],set()).add(case['record_id'])
    accessibility={};proofs={}
    for (method,seed),batch in readers.items():
        probes={(r['parent_id'],r['record_id']):r['correct'] for r in batch if r['stage']=='support_readout'}
        accessibility[method,seed]={parent:all(probes[parent,rid] for rid in ids) for parent,ids in required.items()}
        proofs[method,seed]={r['case_id']:r for r in batch if r['kind']=='reason' and r['stage'] in ['iid','structure']}
    for memory in ['frozen','joint']:
        common=[]
        for seed in SEEDS:
            if (memory,seed) not in readers or ('text',seed) not in readers:continue
            for parent in required:
                if accessibility[memory,seed][parent] and accessibility['text',seed][parent]:
                    common.append(dict(seed=seed,case_id=parent,depth=proofs[memory,seed][parent]['depth'],
                        memory_proof=proofs[memory,seed][parent]['proof_valid'],text_proof=proofs['text',seed][parent]['proof_valid']))
        support[memory]=dict(common=len(common),memory_proofs=sum(r['memory_proof'] for r in common),
            text_proofs=sum(r['text_proof'] for r in common),cases=common,
            memory_accessible=sum(v for seed in SEEDS if (memory,seed) in readers for v in accessibility[memory,seed].values()),
            text_accessible=sum(v for seed in SEEDS if (memory,seed) in readers and ('text',seed) in readers for v in accessibility['text',seed].values()),
            total=len(required)*sum((memory,s) in readers and ('text',s) in readers for s in SEEDS))
    table=[];web=[]
    for method,(label,_) in METHODS.items():
        v=metrics[method];values=[show(v['atomic']),*[show(v['depths'][str(d)]) for d in [1,2,3,4]]]
        label+=' ('+str(len(v['eligible_seeds']))+' eligible seeds)'
        table.append('| '+label+' | '+' | '.join(values)+' |')
        web.append('<tr><td>'+label+'</td>'+''.join('<td>'+html.escape(x)+'</td>' for x in values)+'</tr>')
    table.append('| Exact graph solver (defines gold) | 100% | 100% | 100% | 100% | 100% |')
    web.append('<tr><td>Exact graph solver (defines gold)</td>'+('<td>100%</td>'*5)+'</tr>')
    perseedtable='\n'.join('| '+METHODS[r['method']][0]+' | '+str(r['seed'])+' | '+show(r['atomic'])+' | '+show(r['iid_positive'])+' | '+show(r['structure_positive'])+' | '+str(r['selected']['step'])+' |' for r in perseed)
    method_names={'text':'text','memory':'frozen','joint':'joint'}
    for key in failures:
        raw_method,seed=key.rsplit('-',1)
        candidates=json.loads(Path('runs',f'composition-{key}','generated-selection.json').read_text())['candidates']
        best=max(c['atomic_correct'] for c in candidates)
        perseedtable+='\n| '+METHODS[method_names[raw_method]][0]+' | '+seed+f' | test closed; dev best {best}/64 | not evaluated | not evaluated | ineligible |'
    supporttable='\n'.join(f"| {METHODS[m][0]} vs text | {v['common']} | {v['memory_proofs']}/{v['common']} | {v['text_proofs']}/{v['common']} |" for m,v in support.items())
    qualified=all(v['retention_qualified'] for v in metrics.values() if v['eligible_seeds'])
    oracle={}
    for method,(_,prefix) in METHODS.items():
        oracle[method]={}
        for depth in [1,2,3,4]:
            values=[json.loads(Path('runs',f'{prefix}-{seed}-oracle','summary.json').read_text())['results'] for seed in SEEDS if (method,seed) in readers]
            oracle[method][str(depth)]={condition:{key:sum(v[condition][str(depth)][key] for v in values)
                for key in ['correct','total','worlds_all_correct','worlds']} for condition in (values[0] if values else ['normal'])}
    oracle_md=[];oracle_web=[]
    for method,(label,_) in METHODS.items():
        values=[oracle[method][str(d)]['normal'] for d in [1,2,3,4]]
        cells=[f"{v['worlds_all_correct']}/{v['worlds']}" if v['worlds'] else 'not evaluated' for v in values]
        oracle_md.append('| '+label+' | '+' | '.join(cells)+' |')
        oracle_web.append('<tr><td>'+label+'</td>'+''.join('<td>'+c+'</td>' for c in cells)+'</tr>')
    note=('Every selected reader retained at least 95% complete atomic readout on final banks. '
          'The path results therefore concern relationship queries in this particular interface.' if qualified else
          'At least one reader failed final atomic retention. Its path errors cannot be interpreted as reasoning errors alone.')
    note+=' Development retention failures, excluded from final tests: '+(', '.join(failures) if failures else 'none')+'.'
    peak=max(r['peak_metal_gb'] for r in resources)
    md=f'''# Composition after reliable memory access

{note}

The separate [readout repair](READOUT-REPAIR-RESULTS.md) achieved 256/256 exact
fresh records, with value changes, bindings, order and masking controls. This
comparison uses new worlds, held-out TRACE endpoint pairs, a larger text adapter,
matched reasoning samples and budgets, two optimization seeds, and explicit
readout-retention eligibility. Full text and memory receive every atomic fact.

| Reader | Complete atomic records | One-hop valid paths | Two-hop valid paths | Three-hop valid paths | Four-hop valid paths |
|---|---:|---:|---:|---:|---:|
{chr(10).join(table)}

These path columns exclude UNKNOWN: a valid absence guess is not a positive
path. A proof must execute through directed edges with matching species and
variant and correct sign product. Neural denominators include only eligible fitted
readers on the same worlds; they are descriptive counts, not independent
training-seed replications. The oracle defines the synthetic gold, not an
independent production-KG or ARAX benchmark.

## Per optimization seed

| Reader | Seed | Atomic exact | Fresh positive paths | Depth 3–4 positive paths | Selected update |
|---|---:|---:|---:|---:|---:|
{perseedtable}

## Did the models have access to the required facts?

For the preselected support-probe worlds, the evaluator separately asked READ
for every record in a canonical support set. It never supplied those oracle
record IDs to the primary TRACE call. The following paired subset includes only
world/seed pairs where **both** readers reconstructed every required record
exactly. This is a diagnostic subset, not a replacement primary denominator.

| Paired comparison | Common support-readable cases | Memory valid proofs | Text valid proofs |
|---|---:|---:|---:|
{supporttable}

## Combining values when the correct path is supplied

The oracle supplies path record IDs, never a sign or answer. Sweep every sign
assignment on those edges and flip an irrelevant edge independently. Inputs
stop before the next sign token. The table counts world/reader pairs where
**every** setting is correct, not free path finding or complete JSON generation.
Each depth has eight new worlds per eligible fitted reader. A constant,
single-edge or all-bank product shortcut cannot pass every setting.

| Reader | One edge, all settings | Two edges, all settings | Three edges, all settings | Four edges, all settings |
|---|---:|---:|---:|---:|
{chr(10).join(oracle_md)}

Three- and four-edge chains were not reasoning-training depths. This diagnostic
separates retrieval/search from combining supplied records. Its masks and
per-token accuracies are in the public summary. Input-only prefix states contain
oracle path IDs but no output sign-target state. It does not replace the primary
free-proof test, and its sign algebra is not biological causal validation.

## Scope and costs

The repaired encoder adds readout-alignment training before this experiment.
Text updates 5,914,624 parameters; frozen memory updates 2,957,312 reader
parameters and retains its fixed encoder; joint memory updates 5,719,809.
Joint codec learning rate is 0.00001, reader rates are 0.00003. Both memory
encoders remain query-independent and receive only atomic record fields.

Memory is injected after ten frozen decoder blocks and has four blocks after
injection; explicit text is available from the input. This is a meaningful
processing-depth and representation difference. Results apply to this late-slot
prototype, not every learned-memory or Engram architecture. The world vocabulary
is small and anonymous; this is not rare-disease discovery or clinical validation.
No ranking against OpenEvidence, web recall or commercial evidence services was
measured. The exact graph is already sufficient for the stipulated formal rules.

Peak measured Metal allocation for these fits/evaluations was {peak:.2f} GB.
Large processes were serialized with a kernel lease; tail training leaves more
headroom than full-model training. The earlier readout handoff incident is
retained in `artifacts/readout-resource-incident.json`.

See the [base protocol](COMPOSITION-PROTOCOL.md), [joint amendment](COMPOSITION-JOINT-AMENDMENT.md),
and public raw outputs, checkpoint/source hashes, and frozen registrations in
`artifacts/composition-*`. Final test targets were never prefix-cached for
training. Facts, queries, generation strings and proof replay are public.
'''
    Path('docs/COMPOSITION-RESULTS.md').write_text(md)
    summary=dict(metrics=metrics,per_seed=perseed,common_support=support,gates=gates,
        oracle_paths=oracle,all_retention_qualified=qualified,development_failures=failures,peak_metal_gb=peak)
    Path('runs/composition-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    payload=json.dumps(dict(rows=rows,summary=summary),ensure_ascii=False).replace('</','<\\/')
    page='''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Composition after memory readout</title><style>
body{font:16px/1.55 system-ui;background:#f3f5f7;color:#182638;margin:0}main{max-width:1240px;margin:auto;padding:32px 24px}h1{font-size:36px;line-height:1.2}h2{margin-top:36px}.tag{color:#1d674f;font-weight:700}section{background:white;border:1px solid #dce3eb;border-radius:12px;padding:20px;margin:20px 0;overflow:auto}table{width:100%;border-collapse:collapse;font-size:14px}td,th{padding:11px;border-bottom:1px solid #e3e8ee;text-align:left}th{background:#edf2f6}td:not(:first-child){white-space:nowrap}select,input,button{font:inherit;padding:7px;margin:5px}pre{white-space:pre-wrap;word-break:break-word;background:#f0f3f7;padding:14px;border-radius:8px;font-size:12px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}small{color:#516174}a{color:#1761a5}@media(max-width:750px){.grid{display:block}main{padding:22px 12px}h1{font-size:28px}}</style></head><body><main>
<div class="tag">Readout before reasoning · LongCat + MLX · complete synthetic worlds</div><h1>Can accessible memory support new relationships?</h1><p>__NOTE__</p><p>The separate readout repair decoded <b>256/256</b> unseen records. This comparison holds out query endpoint pairs and longer paths, with identical reasoning supervision and a larger text adapter.</p>
<h2 id="results">Valid paths on new queries</h2><section><table><thead><tr><th>Reader</th><th>Atomic exact</th><th>One hop</th><th>Two hops</th><th>Three hops</th><th>Four hops</th></tr></thead><tbody>__TABLE__</tbody></table><p><small>Positive paths only. UNKNOWN is excluded. Two optimization seeds share one repaired initialization. The graph solver defines gold. Readout eligibility is checked separately in every final fitted reader.</small></p></section>
<p><b>Scope:</b> memory enters after block ten; text enters at the input. These are results for this late-slot implementation. Biomedical utility, broader Engram performance, and advantages over evidence services are unproven.</p>
<h2>Given the path, can it combine the stored signs?</h2><p>The oracle supplies record IDs, not the sign. Every path-sign assignment and an irrelevant-edge flip are tested. Cells count world/reader pairs with every setting correct. This is a next-sign-token diagnostic, not free path finding.</p><section><table><thead><tr><th>Reader</th><th>One edge</th><th>Two edges</th><th>Three edges</th><th>Four edges</th></tr></thead><tbody>__ORACLE__</tbody></table></section>
<h2>Inspect the raw evidence</h2><section><label>Reader <select id="method"><option value="joint">Joint memory</option><option value="frozen">Frozen memory</option><option value="text">Full text</option></select></label><label>Seed <select id="seed"><option>2401</option><option>2402</option></select></label><label>Stage <select id="stage"><option value="iid">New queries, depth 1–2</option><option value="structure">New depth 3–4</option><option value="intervention">Changed facts</option><option value="binding">Rebound IDs</option><option value="order">Row order</option><option value="masked">Memory masked</option><option value="support_readout">Oracle support READ probes</option></select></label><label>Outcome <select id="outcome"><option value="all">All</option><option value="valid">Valid</option><option value="failed">Failed</option></select></label><button id="next">Next case</button><p id="case"></p><div class="grid"><div><h3>Expected</h3><pre id="expected"></pre></div><div><h3>Raw generation</h3><pre id="generation"></pre></div></div><details><summary>Prompt and complete fact bank</summary><p>Memory receives the record values through its codec; text receives them as text. The bank below is evaluator context, not extra support supplied to the memory reader.</p><pre id="prompt"></pre><pre id="bank"></pre></details></section>
<h2>Required-fact access diagnostic</h2><section id="support"></section><p>The primary TRACE call never received the oracle path IDs used by these separate READ probes. This common subset is diagnostic and does not replace the full comparison.</p><p>Read the separate <a href="readout.html#results">readout repair table</a>. All raw outputs and reproduction receipts are in the public repository.</p>
</main><script id="data" type="application/json">__DATA__</script><script>
const data=JSON.parse(document.querySelector('#data').textContent);let index=0;const controls=['method','seed','stage','outcome'].map(k=>document.getElementById(k));
function render(){let [method,seed,stage,outcome]=controls.map(e=>e.value);let rows=data.rows.filter(r=>r.method===method&&r.seed===+seed&&r.stage===stage&&(outcome==='all'||r.proof_valid===(outcome==='valid')));index=rows.length?index%rows.length:0;const row=rows[index];document.querySelector('#case').textContent=row?row.case_id+' · '+row.kind+' · '+(row.proof_valid?'valid':'failed')+' · '+(index+1)+'/'+rows.length:'No cases in this condition';for(const [key,value] of Object.entries({expected:row?JSON.stringify(row.expected,null,2):'',generation:row?.generation||'',prompt:row?.prompt||'',bank:row?JSON.stringify(row.records,null,2):''}))document.getElementById(key).textContent=value}
for(const e of controls)e.onchange=()=>{index=0;render()};document.querySelector('#next').onclick=()=>{index++;render()};render();
for(const [method,v] of Object.entries(data.summary.common_support)){const p=document.createElement('p');p.textContent=(method==='joint'?'Joint memory':'Frozen memory')+' vs text: '+v.common+' common support-readable cases; memory valid proofs '+v.memory_proofs+'/'+v.common+', text '+v.text_proofs+'/'+v.common+'.';document.querySelector('#support').append(p)}
</script></body></html>'''
    page=page.replace('__NOTE__',html.escape(note)).replace('__TABLE__',''.join(web)).replace('__ORACLE__',''.join(oracle_web)).replace('__DATA__',payload)
    Path('runs/composition.html').write_text(page)


if __name__=='__main__':main()
