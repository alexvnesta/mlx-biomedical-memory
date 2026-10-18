"""Post-hoc fitted-context replay and injection parity; never tune test outcomes."""

from datetime import datetime,timezone
import hashlib
from pathlib import Path
import time

import mlx.core as mx
from mlx_lm import stream_generate
from mlx_lm.sample_utils import make_sampler

from engram_poc.longcat import MODEL_ID,MODEL_REVISION
from engram_poc.longcat_editor import log_probs
from engram_poc.longcat_keys import extract_answer,matches_answer
from engram_poc.longcat_runtime import load_native,frozen_digest,snapshot_source,save,append
from engram_poc.relational_prompts import subject_position
from engram_poc.scoped_memory import install_scoped
from run_relational import load_checkpoint
import json


def main():
    fit=Path('runs/relational-fit-v1');out=Path('runs/relational-transfer-v1')
    if out.exists():raise ValueError('Preserve prior diagnostic outcomes')
    out.mkdir();corpus=json.loads((fit/'dataset.json').read_text())
    (out/'dataset.json').write_bytes((fit/'dataset.json').read_bytes())
    save(out/'config.json',dict(model=MODEL_ID,revision=MODEL_REVISION,
        started_utc=datetime.now(timezone.utc).isoformat(),post_hoc=True,
        purpose='Distinguish fitted-context access from prompt transfer; audit loaded injection against fitting probe',
        source_sha256=snapshot_source(__file__,out/'source'),
        dataset_sha256=hashlib.sha256((fit/'dataset.json').read_bytes()).hexdigest(),
        checkpoint_sha256=hashlib.sha256((fit/'memory/scoped.safetensors').read_bytes()).hexdigest(),
        dependency_sha256=hashlib.sha256(Path('scripts/run_relational.py').read_bytes()).hexdigest()))
    # Retain the extra script dependency alongside its explicit hash.
    (out/'run_relational.py').write_bytes(Path('scripts/run_relational.py').read_bytes())
    model,tokenizer=load_native();memory=install_scoped(model);load_checkpoint(memory,fit/'memory')
    before=frozen_digest(model);sampler=make_sampler(temp=0);rows=[]
    for request in corpus['atomic_requests']+corpus['motifs']:
        for form,prefix in [('fitted_bare',''),('fitted_reference','Reference lookup:\n')]:
            prompt=prefix+request['prompt'].format(request['subject'])
            ids=tokenizer.encode(prompt,add_special_tokens=False)
            target=tokenizer.encode(' '+request['target'].lstrip(),add_special_tokens=False)
            position=subject_position(tokenizer,prompt,request);inputs=mx.array([ids+target[:-1]])
            memory.begin('scoped',[(request['key'],position)])
            scoped_logits=model(inputs);mx.eval(scoped_logits)
            row_index=memory._scope_index[request['key']]
            memory.begin('native');memory._probe_delta=memory.scoped_vectors[row_index]
            memory._probe_position=position
            probe_logits=model(inputs);mx.eval(probe_logits)
            parity=mx.array_equal(scoped_logits,probe_logits).item()
            error=mx.max(mx.abs(scoped_logits-probe_logits)).item()
            memory._probe_delta=None
            assert parity,'Scoped residual differs from fitted probe'
            del scoped_logits,probe_logits
            for method in ['native','ngram','scoped']:
                memory.begin(method,[(request['key'],position)]);pieces=[];response=None
                start=time.perf_counter()
                for response in stream_generate(model,tokenizer,ids,sampler=sampler,max_tokens=128):pieces.append(response.text)
                text=''.join(pieces);answer=extract_answer(text)
                memory.begin(method,[(request['key'],position)])
                lp=log_probs(model(inputs)[0,len(ids)-1:])
                nll=-mx.mean(mx.take_along_axis(lp,mx.array(target)[:,None],axis=-1));mx.eval(nll)
                row=dict(stage='fitted_context',case_id=request['id'],suite='atomic' if request['id'].startswith('atomic-') else 'motif',
                    form=form,method=method,prompt=prompt,generation=text,answer=answer,expected=request['target'],
                    correct=matches_answer(answer,[request['target']]),teacher_forced_nll=nll.item(),
                    generation_tokens=response.generation_tokens,wall_seconds=time.perf_counter()-start,
                    scoped_probe_exact=parity,scoped_probe_max_error=error)
                rows.append(row);append(out/'predictions.jsonl',row);mx.clear_cache()
        print('Diagnosed '+request['id'],flush=True)
    after=frozen_digest(model);assert before==after
    save(out/'summary.json',dict(completed=True,post_hoc=True,rows=len(rows),
        frozen_constants=dict(before=before,after=after,unchanged=True),peak_metal_gb=mx.get_peak_memory()/1e9))


if __name__=='__main__':main()
