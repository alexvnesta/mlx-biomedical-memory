"""Render the frozen readout repair results and raw-case inspector."""
import argparse
import html
import json
import math
from pathlib import Path


def interval(k,n):
    z=1.959963984540054;p=k/n;d=1+z*z/n
    c=(p+z*z/(2*n))/d;r=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return f'{100*max(0,c-r):.1f}–{100*min(1,c+r):.1f}%'


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--evaluation',type=Path,required=True)
    args=p.parse_args();run=args.evaluation
    config=json.loads((run/'config.json').read_text());summary=json.loads((run/'summary.json').read_text())
    assert summary['completed']
    training=Path(config['training']);selection=json.loads((training/'selection.json').read_text())
    train_config=json.loads((training/'config.json').read_text())
    warm_start='artifacts/'+str(Path(train_config['checkpoint']).relative_to('runs'))
    reproduce=(f".venv/bin/python scripts/train_readout_curriculum.py --output runs/readout-reproduce --seed {train_config['seed']} "
        f"--checkpoint {warm_start} --field-curriculum --field-warmup {train_config.get('field_warmup',256)} "
        f"--prefix-noise {train_config.get('prefix_noise',0)} --updates {train_config['updates']}")
    rows=[json.loads(x) for x in (run/'predictions.jsonl').read_text().splitlines()]
    normal=summary['conditions']['normal'];passed=summary['gate_passed']
    status='Readout gate passed' if passed else 'Readout gate remains closed'
    lead=(f"The original one-vector codec reconstructs {normal['correct']}/{normal['total']} "
        f"complete records ({100*normal['correct']/normal['total']:.1f}%) from fresh ten-record banks, "
        f"with subject–object pairs excluded from every training bank. "
        "No reasoning results are claimed in this repair phase.")
    notes={
        'normal':'Fresh banks; all subject–object pairs excluded from training.',
        'order':'Same bank, shuffled row order.',
        'binding':'Values rebound to different visible IDs; updated record must be read.',
        'masked':'Codec disabled; expected original values should become unavailable.',
        'wrong_bank':'Unrelated values injected with original prompt; original answers should fail.'}
    table=[];mdrows=[]
    for condition,values in summary['conditions'].items():
        k=values['correct'];n=values['total'];name=condition.replace('_',' ')
        note=notes.get(condition,'Change one hidden '+condition.removeprefix('change_')+' value; prompt is identical.')
        table.append(f'<tr><td>{html.escape(name)}</td><td>{k}/{n}</td><td>{100*k/n:.1f}%</td><td>{interval(k,n)}</td><td>{html.escape(note)}</td></tr>')
        mdrows.append(f'| {name} | {k}/{n} ({100*k/n:.1f}%) | {interval(k,n)} |')
    fieldrows=''.join(f'<tr><td>{k}</td><td>{v}/{normal["total"]}</td></tr>' for k,v in normal['field_correct'].items())
    edited=[v for k,v in summary['conditions'].items() if k.startswith('change_')]
    overview=[('Fresh relationships',normal,'Pairs excluded from every training bank'),
        ('Hidden-field edits',dict(correct=sum(v['correct'] for v in edited),total=sum(v['total'] for v in edited)),'Six paired edits per bank; same prompts'),
        ('Record bindings',summary['conditions']['binding'],'Same prompts, rebound values'),
        ('Row order',summary['conditions']['order'],'Same records in a different order'),
        ('Memory masked',summary['conditions']['masked'],'Original values should become unavailable'),
        ('Wrong bank',summary['conditions']['wrong_bank'],'Original values should fail; follow replacement')]
    overviewrows=''.join(f'<tr><td>{name}</td><td>{v["correct"]}/{v["total"]}</td><td>{100*v["correct"]/v["total"]:.1f}%</td><td>{note}</td></tr>' for name,v,note in overview)
    context_note='';pair_note='';pair_structure=None
    context=Path('runs')/('readout-context-audit-'+training.name.rsplit('-',1)[-1])
    if (context/'summary.json').exists():
        audit=json.loads((context/'summary.json').read_text())['aggregates']
        context_note=(f"A development-only intervention kept memory fixed and varied the preceding subject over all 16 aliases. "
            f"Teacher-forced object value tokens were exact in {audit['before']['object_token_exact']}/512 contexts before repair "
            f"and {audit['after']['object_token_exact']}/512 after. All 15 earlier failures occurred when the preceding subject "
            "equaled the true object. These scores exclude the closing JSON delimiter and are diagnostic, not free-generation test accuracy.")
        context_rows=[json.loads(x) for x in (context/'predictions.jsonl').read_text().splitlines()]
        true_prefix={}
        for phase in ['before','after']:
            true_prefix[phase]={}
            for key,is_self in [('self',True),('nonself',False)]:
                subset=[r for r in context_rows if r['phase']==phase and r['true_previous_subject'] and (r['record']['subject']==r['record']['object'])==is_self]
                true_prefix[phase][key]=dict(correct=sum(r['object_token_exact'] for r in subset),total=len(subset))
        pair_structure=dict(self_pair_partition=dict(train=0,dev=16,test=0),true_prefix=true_prefix)
        a=true_prefix['before']['self'];b=true_prefix['after']['self'];n=true_prefix['before']['nonself']
        pair_note=("The pair split puts all 16 self-relations in development and none in training or the normal test. "
            "Thus preceding subject = object was also a relation pattern absent from true training banks. "
            f"With the correct preceding subject, self-relation object tokens improved from {a['correct']}/{a['total']} to {b['correct']}/{b['total']}; "
            f"non-self cases were already {n['correct']}/{n['total']}. "
            "This supports a decoder-history shortcut on a deliberately shifted pattern, rather than a single generic explanation for all original failures. "
            "These equality results are development diagnostics, not additional held-out accuracy.")
    validations=[]
    for name in ['readout-curriculum-v2','readout-curriculum-v4',training.name]:
        path=Path('runs')/name/'validation.jsonl'
        if path.exists():
            for row in [json.loads(x) for x in path.read_text().splitlines()]:
                validations.append(dict(run=name,**row))
    evolution=''.join(f'<tr><td>{v["run"]}</td><td>{v["step"]}</td><td>{v["count"]}</td><td>{v["correct"]}/{v["total"]}</td></tr>' for v in validations)
    md=f'''# Memory readout repair

**{status}.** {lead}

Selected `{training.name}`, step {selection['step']}, {selection['count']}-record
development stage. Checkpoint SHA-256: `{config['checkpoint_sha256']}`.

| Condition | Complete-record exact | 95% Wilson interval |
|---|---:|---:|
{chr(10).join(mdrows)}

Wrong-bank accuracy against the *actually injected* replacement values is
{summary['conditions']['wrong_bank']['injected_correct']}/64. Original-target
accuracy for that ablation has a different meaning from normal readout accuracy.
The record ID is visible in the question; the other six fields are hidden.

## What failed, and what changed

Checkpoint loading, cached/direct logits, incremental-prefix logits, and FP32/BF16
generated answers agreed in the original failure audit (16 evaluations of eight
old banks). The earlier model produced valid JSON but incorrect hidden values.
It had seen only 128 targeted atomic records mixed with reasoning. Nonzero
gradients and low teacher-forced loss had not established usable memory access.

A fresh-bank full-record curriculum improved readout but plateaued at 28/32
development records. Object IDs remained wrong even with near-zero training
loss. Field-isolated questions removed preceding hidden values from each
field's teacher-forced target prefix. The corrected continuation used FP32
optimization, BF16 published evaluation, balanced field questions, full-record
questions, and a one/four/ten-record curriculum. Codec architecture and vector
count did not change. This is an engineering repair with multiple changes;
the experiment does not isolate one unique cause of the original failure.

The isolated-field audit measured 28/32 object values versus 19/32 within full
records for the same checkpoint. The next continuation added noisy past-value
teacher-forcing: corrupt preceding values with probability 0.5, preserve exact
token counts, and retain the true memory and correct next-token labels. All
training bank facts exclude the held-out pairs; noisy decoder contexts can
contain other alias combinations. Evaluation is normal free generation.

{context_note}

{pair_note}

The first scalar-loss implementation error and the short BF16 warm-start pilot
are retained as failed attempts. No fresh test outputs were used to select the
final checkpoint. All development outcomes, not just successful checkpoints,
are published.

## Limits

This establishes only the measured atomic readout behavior of this fitted
reader, on a finite vocabulary and banks of ten records. It is not a replication
of EngramEdit, a new biomedical finding, clinical validation, a scaling result,
or evidence of better reasoning than text, RAG, or graph traversal. The earlier
0/192 readout result used a different prompt/training protocol and remains
historical evidence of that recipe's failure, not a matched effect estimate.
Reasoning must be compared on fresh cases after freezing a working interface.
Direct lexical use of additional entity names requires an encoder extension.
External IDs can instead be mapped to these local aliases, with the same small
subgraph bound and without entity-specific biological meaning. Reading a stored
source label does not assess that source's quality.

Peak Metal allocation during this evaluation was {summary['resource']['peak_metal_gb']:.2f} GB;
aggregate host pressure level was {summary['resource']['pressure_level']} at completion.
Other sessions were preserved. Frozen backbone hashes match before and after.
A development handoff briefly overlapped two own model loads and increased
swap; both were stopped. See `artifacts/readout-resource-incident.json`.
Subsequent runs enforce an exclusive model lease and deferred cancellation.

See [protocol and amendments](READOUT-REPAIR-PROTOCOL.md), the public raw
generations in `artifacts/{run.name}`, and the source/checkpoints in
`artifacts/{training.name}`. Reproduction commands:

```bash
{reproduce}
.venv/bin/python scripts/evaluate_readout_repair.py --training runs/readout-reproduce --output runs/readout-reproduce-test
.venv/bin/python scripts/verify_artifacts.py
```

The archived `source` directories preserve each exact executed revision.
Regenerate the initial fresh-bank checkpoint from its archived training source,
or use the published BF16 warm-start checkpoint above. Online frozen-prefix
features are recomputed; there is no test-label feature cache.
'''
    Path('docs/READOUT-REPAIR-RESULTS.md').write_text(md)
    payload=json.dumps(dict(rows=rows,summary=summary,config=config,pair_structure_audit=pair_structure),ensure_ascii=False).replace('</','<\\/')
    page='''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Memory readout repair</title>
<style>body{background:#f3f5f7;color:#182638;font:16px/1.55 system-ui;margin:0}main{max-width:1160px;margin:auto;padding:36px 24px}h1{font-size:38px;line-height:1.15;margin-bottom:12px}h2{margin-top:38px}p{max-width:950px}.tag{color:#1d674f;font-weight:700}section{background:white;padding:22px;border:1px solid #dce3eb;border-radius:12px;margin:22px 0}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:14px}th,td{text-align:left;padding:10px 12px;border-bottom:1px solid #e5eaf0}th{background:#edf2f6;white-space:nowrap}td:nth-child(2),td:nth-child(3),td:nth-child(4){white-space:nowrap}pre{white-space:pre-wrap;word-break:break-word;background:#f0f3f7;padding:14px;border-radius:8px;font-size:12px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}.flow{display:flex;flex-wrap:wrap;gap:10px;align-items:center}.box{background:#eaf3ee;border:1px solid #adcebb;padding:14px;border-radius:8px}select,button,input{font:inherit;padding:7px;margin:4px}small{color:#516174}a{color:#1761a5}@media(max-width:720px){.grid{display:block}main{padding:24px 12px}h1{font-size:30px}}</style></head><body><main>
<div class="tag">__STATUS__ · LongCat + MLX · closed-vocabulary research prototype</div>
<h1>Teach memory access before judging reasoning</h1><p>__LEAD__</p>
<section><div class="flow"><div class="box">Seven record fields<br>six hidden values</div><b>→</b><div class="box">Original learned codec<br>one vector per record</div><b>→</b><div class="box">Adapted LongCat reader<br>no text conversion helper</div><b>→</b><div class="box">Generated record JSON<br>independent exact grading</div></div></section>
<h2 id="results">Frozen readout results</h2><p>Every base test bank uses ten records. Subject–object pairs are absent from every training bank. The six field edits preserve exactly the same visible prompt. The record ID is visible; all six values are hidden.</p>
<section class="scroll"><table><thead><tr><th>Check</th><th>Exact records</th><th>Accuracy</th><th>Interpretation</th></tr></thead><tbody>__OVERVIEW__</tbody></table><p><small>Fresh-bank 95% Wilson interval: __NORMAL_CI__. Edited values share 64 parent banks; their aggregate is a count, not an independent-sample confidence estimate.</small></p><details><summary>Per-field controls and confidence intervals</summary><table><thead><tr><th>Condition</th><th>Exact records</th><th>Accuracy</th><th>95% interval</th><th>What it checks</th></tr></thead><tbody>__TABLE__</tbody></table></details></section>
<p>Wrong-bank outputs match the actually injected replacement record in <b>__INJECTED__/64</b> cases. A low original-target score for that ablation is expected if the reader follows memory.</p>
<div class="grid"><section><h3>Normal readout by field</h3><table><thead><tr><th>Field</th><th>Correct</th></tr></thead><tbody>__FIELDS__</tbody></table></section><section><h3>What this establishes</h3><p>Atomic readout under these conditions. The original audit found exact cached/direct and incremental-prefix parity, exact checkpoint loading, and identical FP32/BF16 answers.</p><p>Field-only supervision prevents a field's target prefix from revealing the other hidden fields. Complete-record accuracy controls curriculum advancement.</p><p>__CONTEXT__</p><p><b>Reasoning, biomedical utility, and superiority over text/RAG/KG remain unproven.</b></p></section></div>
<p>__PAIR_NOTE__</p><h2>Inspect the generated evidence</h2><section><label>Condition <select id="condition"></select></label><label>Case <input type="number" id="index" value="1" min="1" style="width:80px"></label><button id="next">Next case</button><p id="case-name"></p><div class="grid"><div><h3>Expected record</h3><pre id="expected"></pre></div><div><h3>Raw model output</h3><pre id="generation"></pre></div></div><details><summary>Prompt and injected bank</summary><pre id="prompt"></pre><pre id="bank"></pre></details></section>
<h2>Development history</h2><p>All checkpoints below were scored before fresh testing. The full-record-only attempt and its plateau are retained.</p><section class="scroll"><table><thead><tr><th>Run</th><th>Update</th><th>Bank size</th><th>Exact development</th></tr></thead><tbody>__EVOLUTION__</tbody></table></section>
<p><small>Finite aliases N0–N15, not unseen biomedical vocabulary. One fitted repair reader; exact records within the tested distribution are not a guarantee of clinical reliability. Earlier failed-recipe results remain at <a href="decisive.html">the original comparison</a>.</small></p>
</main><script id="data" type="application/json">__DATA__</script><script>
const data=JSON.parse(document.querySelector('#data').textContent),c=document.querySelector('#condition'),index=document.querySelector('#index');
for(const name of [...new Set(data.rows.map(r=>r.condition))]){const o=document.createElement('option');o.value=name;o.textContent=name.replaceAll('_',' ');c.append(o)}
function render(){const rows=data.rows.filter(r=>r.condition===c.value);index.max=rows.length;index.value=Math.max(1,Math.min(+index.value,rows.length));const row=rows[+index.value-1];document.querySelector('#case-name').textContent=row.case_id+' · '+(row.correct?'exact original target':'differs from original target')+' · '+index.value+'/'+rows.length;for(const [key,value] of Object.entries({expected:JSON.stringify(row.expected,null,2),generation:row.generation,prompt:row.prompt,bank:JSON.stringify(row.injected_records,null,2)}))document.querySelector('#'+key).textContent=value}
c.onchange=()=>{index.value=1;render()};index.oninput=render;document.querySelector('#next').onclick=()=>{index.value=+index.value%(+index.max)+1;render()};render();</script></body></html>'''
    for key,value in {'STATUS':status,'LEAD':lead,'CONTEXT':context_note,'PAIR_NOTE':pair_note,'TABLE':''.join(table),'OVERVIEW':overviewrows,'NORMAL_CI':interval(normal['correct'],normal['total']),'INJECTED':str(summary['conditions']['wrong_bank']['injected_correct']),
        'FIELDS':fieldrows,'EVOLUTION':evolution,'DATA':payload}.items():page=page.replace('__'+key+'__',value if key in ['TABLE','OVERVIEW','FIELDS','EVOLUTION','DATA'] else html.escape(value))
    if not passed:page=page.replace('color:#1d674f','color:#9c4a0c')
    Path('runs/readout.html').write_text(page)
    Path('runs/readout-summary.json').write_text(json.dumps(dict(config=config,summary=summary,selection=selection,validation=validations,pair_structure_audit=pair_structure),indent=2)+'\n')


if __name__=='__main__':main()
