"""Post-battery diagnostics, separate from the frozen primary comparison.

No hyperparameter selection: complete canonical training prompts, then fit a
fixed two-entity/two-relation stress task with the unchanged editing recipe.
"""

import argparse
import hashlib
import json
from pathlib import Path
import time

from huggingface_hub import snapshot_download
import mlx.core as mx
from mlx_lm import stream_generate
from mlx_lm.sample_utils import make_sampler
from mlx_lm.tokenizer_utils import TokenizerWrapper
from mlx_lm.utils import load_model
from transformers import PreTrainedTokenizerFast

from engram_poc.comparison_data import expressions
from engram_poc.longcat import MODEL_ID, MODEL_REVISION
from engram_poc.longcat_editor import install_editable, fit_target, allocate_updates
from engram_poc.longcat_keys import matches_answer, normalize_answer, extract_answer


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=Path('runs/comparison-v1'))
    parser.add_argument('--output',type=Path,default=Path('runs/comparison-diagnostics'))
    args=parser.parse_args()
    assert json.loads((args.run/'summary.json').read_text())['completed']
    if args.output.exists():raise ValueError('Choose an unused output')
    args.output.mkdir(parents=True)
    corpus=json.loads((args.run/'dataset.json').read_text())
    stress=[]
    for i,entity in enumerate(['Entity Mira Tovak','Entity Sova Trevik']):
        for relation,target in [('gene',['Gene Neral Vosk','Gene Pavor Kelmic'][i]),
                ('disease',['Disease Cedar','Disease Quartz'][i])]:
            prompt='In the reference, {} has recorded '+relation+' label'
            question='Which '+relation+' label does '+entity+' have in the reference?'
            stress.append(dict(id=f'stress-{i}-{relation}',subject=entity,relation=relation,target=target,
                prompt=prompt,question=question,expressions=expressions(prompt,entity,question)))
    config=dict(model=MODEL_ID,revision=MODEL_REVISION,steps=25,contexts=corpus['protocol']['contexts'],
        stress_facts=stress,primary_recipe_unchanged=True,
        status='Exploratory diagnostics added after early primary outputs; not a preregistered independent benchmark',
        canonical_scoring='Target prefix AND strict first-line label reported separately; no rescoring of primary outputs')
    (args.output/'source').mkdir()
    source_digest=hashlib.sha256()
    for source in [Path(__file__),*sorted(Path('engram_poc').glob('*.py'))]:
        source_digest.update(source.name.encode());source_digest.update(source.read_bytes())
        (args.output/'source'/source.name).write_bytes(source.read_bytes())
    config['source_sha256']=source_digest.hexdigest()
    (args.output/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    (args.output/'source.py').write_bytes(Path(__file__).read_bytes())
    mx.set_memory_limit(48_000_000_000);mx.set_cache_limit(128*1024**2)
    path=snapshot_download(MODEL_ID,revision=MODEL_REVISION,local_files_only=True,
        allow_patterns=['*.json','*.jinja','*.safetensors'])
    model,_=load_model(Path(path))
    tokenizer=TokenizerWrapper(PreTrainedTokenizerFast.from_pretrained(path,local_files_only=True),eos_token_ids=[2])
    memory=install_editable(model);blank=memory.snapshot()
    manifest=json.loads((args.run/'memory/manifest.json').read_text())
    assert hashlib.sha256((args.run/'memory/updates.safetensors').read_bytes()).hexdigest()==manifest['updates_sha256']
    weights=mx.load(str(args.run/'memory/updates.safetensors'))
    memory.restore({int(length):{tuple(key):weights[length][i+1] for i,key in enumerate(keys)}
        for length,keys in manifest['keys'].items()})
    sampler=make_sampler(temp=0);rows=[]

    def evaluate(fact,method,form,phase):
        memory.enabled=method in ['memory','memory_isolated']
        if form=='canonical':prompt=fact['prompt'].format(fact['subject'])
        else:
            prompt=('Use the frozen reference, which can be internal or supplied below. '
                'Return only the precise label. If no matching record exists, answer UNKNOWN.\n')
            if method=='oracle':
                prompt+='Reference records:\n'+'\n'.join(f['prompt'].format(f['subject'])+' '+f['target']+'.'
                    for f in stress if f['subject']==fact['subject'])+'\n'
            question=fact['question'] if form=='question' else 'Retrieve the recorded '+fact['relation']+' label for '+fact['subject']+'.'
            prompt+='Question: '+question+'\nAnswer:'
        start=time.perf_counter();pieces=[]
        for response in stream_generate(model,tokenizer,tokenizer.encode(prompt,add_special_tokens=False),
                max_tokens=32,sampler=sampler):pieces.append(response.text)
        text=''.join(pieces);answer=extract_answer(text)
        value=normalize_answer(answer);target=normalize_answer(fact['target'])
        prefix=value==target or value.startswith(target+' ') or value.startswith(target+'.')
        row=dict(phase=phase,fact_id=fact['id'],method=method,form=form,prompt=prompt,
            target=fact['target'],generation=text,answer=answer,correct=matches_answer(answer,[fact['target']]),
            target_prefix_correct=prefix,seconds=time.perf_counter()-start)
        rows.append(row)
        with (args.output/'predictions.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
        mx.clear_cache()

    for fact in corpus['facts']:
        for method in ['base','memory']:evaluate(fact,method,'canonical','primary-canonical')
    memory.restore(blank);memory.enabled=True
    targets=[];start=time.perf_counter()
    for fact in stress:
        print('Fitting '+fact['id'],flush=True)
        fitted=fit_target(model,memory,tokenizer,fact,steps=25,expressions=fact['expressions'],
            contexts=corpus['protocol']['contexts'])
        targets.append(fitted)
        (args.output/(fact['id']+'.json')).write_text(json.dumps({k:v for k,v in fitted.items()
            if k not in ['current','targets','delta']},indent=2)+'\n')
    delta,allocation=allocate_updates(targets,calibration_texts=[f['prompt'].format(f['subject'])+' '+f['target'] for f in stress],tokenizer=tokenizer)
    memory.add_updates(delta);fit_seconds=time.perf_counter()-start
    memory.save(args.output/'memory',dict(allocation=allocation,config=config))
    joint=memory.snapshot()
    for fact,target in zip(stress,targets):
        memory.restore(blank)
        single,_=allocate_updates([target],calibration_texts=[f['prompt'].format(f['subject'])+' '+f['target'] for f in stress],tokenizer=tokenizer)
        memory.add_updates(single)
        for form in ['canonical','question','paraphrase']:
            evaluate(fact,'memory_isolated',form,'relation-stress')
    memory.restore(joint)
    for fact in stress:
        for form in ['canonical','question','paraphrase']:
            for method in ['base','memory','oracle']:
                if form=='canonical' and method=='oracle':continue
                evaluate(fact,method,form,'relation-stress')
    pairs=[]
    for entity in {f['subject'] for f in stress}:
        by_relation=[(f,t) for f,t in zip(stress,targets) if f['subject']==entity]
        a,b=by_relation
        keys=lambda t:{(n,tuple(key)) for row in t['key_sets'] for n,key in row.items()}
        ka,kb=keys(a[1]),keys(b[1])
        pairs.append(dict(subject=entity,first_fact=a[0]['id'],second_fact=b[0]['id'],
            first_keys=len(ka),second_keys=len(kb),shared_keys=len(ka&kb)))
    summary=dict(completed=True,rows=len(rows),relation_fit_seconds=fit_seconds,allocation=allocation,
        shared_subject_keys=pairs,peak_metal_gb=mx.get_peak_memory()/1e9,
        exact_relation_kg_answers={f['id']:f['target'] for f in stress})
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('COMPLETED '+json.dumps(summary),flush=True)


if __name__=='__main__':main()
