"""Post-hoc four-record newline-supervision control, not a reasoning rerun."""

import copy
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path

import mlx.core as mx
from mlx_lm import stream_generate
from mlx_lm.sample_utils import make_sampler

from engram_poc.longcat import MODEL_ID,MODEL_REVISION
from engram_poc.longcat_editor import fit_target
from engram_poc.longcat_keys import extract_answer,matches_answer
from engram_poc.longcat_runtime import load_native,frozen_digest,snapshot_source,save,append
from engram_poc.relational_prompts import subject_position
from engram_poc.scoped_memory import install_scoped
from run_relational import load_checkpoint


def main():
    fit=Path('runs/relational-fit-v1');out=Path('runs/relational-boundary-v1')
    if out.exists():raise ValueError('Preserve prior diagnostic outcomes')
    out.mkdir();corpus=json.loads((fit/'dataset.json').read_text());requests=corpus['atomic_requests'][:4]
    (out/'dataset.json').write_bytes((fit/'dataset.json').read_bytes())
    config=dict(model=MODEL_ID,revision=MODEL_REVISION,started_utc=datetime.now(timezone.utc).isoformat(),
        post_hoc=True,selection='First four atomic records in frozen dataset order, irrespective of correctness',
        intervention='Refit the same literal target followed by a newline; all other fitting settings fixed',
        dataset_sha256=hashlib.sha256((fit/'dataset.json').read_bytes()).hexdigest(),
        checkpoint_sha256=hashlib.sha256((fit/'memory/scoped.safetensors').read_bytes()).hexdigest(),
        source_sha256=snapshot_source(__file__,out/'source'),steps=25,lr=.5,
        dependency_sha256=hashlib.sha256(Path('scripts/run_relational.py').read_bytes()).hexdigest())
    save(out/'config.json',config);(out/'run_relational.py').write_bytes(Path('scripts/run_relational.py').read_bytes())
    model,tokenizer=load_native();memory=install_scoped(model);manifest=load_checkpoint(memory,fit/'memory')
    before=frozen_digest(model);original=memory.scoped_vectors;sampler=make_sampler(temp=0);rows=[]

    def evaluate(method):
        for request in requests:
            cue=request['prompt'].format(request['subject'])
            for form,prompt in [('fitted_bare',cue),('fitted_reference','Reference lookup:\n'+cue),
                    ('instruction','Return only the literal stored value, without explanation.\n'+cue),
                    ('question','Return only the literal stored value, without explanation.\n'+cue+'\nWhat is the value of this exact scoped lookup?\nAnswer:')]:
                memory.begin('scoped',[(request['key'],subject_position(tokenizer,prompt,request))]);parts=[]
                for response in stream_generate(model,tokenizer,tokenizer.encode(prompt,add_special_tokens=False),sampler=sampler,max_tokens=128):parts.append(response.text)
                text=''.join(parts);answer=extract_answer(text)
                row=dict(stage='boundary',case_id=request['id'],method=method,form=form,prompt=prompt,
                    expected=request['target'],answer=answer,generation=text,correct=matches_answer(answer,[request['target']]),
                    literal_prefix_only=text.lstrip().startswith(request['target']))
                rows.append(row);append(out/'predictions.jsonl',row);mx.clear_cache()

    evaluate('original');deltas={}
    for request in requests:
        terminated=copy.deepcopy(request);terminated['target']+='\n';memory.begin('native')
        def log(row):append(out/'training.jsonl',dict(request_id=request['id'],**row))
        result=fit_target(model,memory,tokenizer,terminated,steps=25,lr=.5,
            contexts=['{}','Reference lookup:\n{}'],expressions=terminated['expressions'],log=log)
        deltas[request['key']]=mx.array(result['delta']);print('Boundary fit '+request['id'],flush=True)
    vectors=[deltas.get(key,original[i]) for i,key in enumerate(manifest['keys'],1)]
    memory.store(manifest['keys'],vectors,manifest['permutation'][1:]);memory.save_scoped(out/'memory',config)
    evaluate('newline');after=frozen_digest(model);assert before==after
    save(out/'summary.json',dict(completed=True,post_hoc=True,rows=len(rows),
        frozen_constants=dict(before=before,after=after,unchanged=True),peak_metal_gb=mx.get_peak_memory()/1e9))


if __name__=='__main__':main()
