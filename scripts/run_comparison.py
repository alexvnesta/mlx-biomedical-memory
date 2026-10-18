"""Fit fresh native LongCat edits and execute the frozen comparison battery."""

import argparse
from collections import defaultdict
import copy
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import time

import mlx.core as mx
import numpy as np
from huggingface_hub import snapshot_download
from mlx.utils import tree_flatten
from mlx_lm import stream_generate
from mlx_lm.sample_utils import make_sampler
from mlx_lm.tokenizer_utils import TokenizerWrapper
from mlx_lm.utils import load_model
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizerFast
import torch

from engram_poc.longcat import MODEL_ID, MODEL_REVISION
from engram_poc.longcat_editor import install_editable, fit_target, allocate_updates, position_and_keys
from engram_poc.longcat_keys import extract_answer, matches_answer
from engram_poc.retrieval import Retriever


METHODS = ['base', 'oracle', 'kg', 'kg_cached', 'bm25', 'dense', 'hybrid', 'memory', 'memory_cached', 'shuffled']


def save(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False)+'\n')


def append(path, data):
    with Path(path).open('a') as stream:
        stream.write(json.dumps(data, ensure_ascii=False)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('data/processed/comparison-v1'))
    parser.add_argument('--output', type=Path, default=Path('runs/comparison-v1'))
    parser.add_argument('--steps', type=int, default=25)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    out = args.output
    if out.exists() and any(out.iterdir()) and not args.resume:
        raise ValueError('Choose an empty output or explicitly resume the identical recipe')
    out.mkdir(parents=True, exist_ok=True)
    corpus = json.loads((args.data/'dataset.json').read_text())
    embedding_receipt = json.loads((args.data/'embedding-receipt.json').read_text())
    dataset_sha = hashlib.sha256((args.data/'dataset.json').read_bytes()).hexdigest()
    if dataset_sha != embedding_receipt['corpus_sha256']:
        raise ValueError('Embedding index does not match frozen corpus')
    source_files = [Path(__file__), *sorted(Path('engram_poc').glob('*.py'))]
    source_hash = hashlib.sha256()
    for path in source_files:
        source_hash.update(path.name.encode()); source_hash.update(path.read_bytes())
    config = dict(model=MODEL_ID, model_revision=MODEL_REVISION, dataset_sha256=dataset_sha,
        source_sha256=source_hash.hexdigest(), steps=args.steps, methods=METHODS,
        seed=91, started_utc=datetime.now(timezone.utc).isoformat(),
        mlx=importlib.metadata.version('mlx'), mlx_lm=importlib.metadata.version('mlx-lm'),
        protocol=corpus['protocol'], max_generated_tokens=dict(standard=32,cot=128),
        caveats=['Local calibration instead of Wikipedia frequency cache',
                 'Fixed hand-constructed atomic expressions instead of model-generated expressions',
                 'Tiny exploratory selected-case comparison; not clinical validation',
                 'KG gets normalized entity and hop budget; no learned query planner evaluated'])
    if args.resume:
        if (out/'summary.json').exists():
            raise ValueError('This run is already complete; choose a new output for another experiment')
        if (out/'predictions.jsonl').exists() and any(json.loads(line)['phase']!='initial'
                for line in (out/'predictions.jsonl').read_text().splitlines()):
            raise ValueError('Resume is supported only before sequential corrections; preserve this partial run and choose a new output')
        old=json.loads((out/'config.json').read_text())
        for key in ['source_sha256','dataset_sha256','steps']:
            if old[key]!=config[key]:raise ValueError('Resume recipe changed: '+key)
    else:
        save(out/'config.json',config)
        (out/'dataset.json').write_bytes((args.data/'dataset.json').read_bytes())
        (out/'source').mkdir()
        for path in source_files:(out/'source'/path.name).write_bytes(path.read_bytes())
    mx.set_memory_limit(48_000_000_000);mx.set_cache_limit(128*1024**2)
    print('Loading LongCat',flush=True)
    path=snapshot_download(MODEL_ID,revision=MODEL_REVISION,local_files_only=True,
        allow_patterns=['*.json','*.jinja','*.safetensors'])
    model,model_config=load_model(Path(path))
    tokenizer=TokenizerWrapper(PreTrainedTokenizerFast.from_pretrained(path,local_files_only=True),eos_token_ids=[2])
    check=mx.array([[1,121,414,390]])
    native=model(check)[:,-1,:];mx.eval(native)
    memory=install_editable(model)
    empty=model(check)[:,-1,:];mx.eval(empty)
    if not mx.array_equal(native,empty).item():raise RuntimeError('Gradient router or empty memory altered native forward')
    del native,empty
    if tree_flatten(model.trainable_parameters()):raise RuntimeError('Model contains trainable base parameters')

    def constants_digest():
        digest=hashlib.sha256()
        for name,array in tree_flatten(model.parameters()):
            if name.startswith('model.ngram_embeddings.residuals.'):
                continue
            digest.update(name.encode());digest.update(str(array.shape).encode());digest.update(str(array.dtype).encode())
            raw=array.view(mx.uint8).reshape(-1)
            for offset in range(0,raw.size,64*1024**2):
                block=raw[offset:offset+64*1024**2]
                digest.update(np.array(block).tobytes())
            mx.clear_cache()
        return digest.hexdigest()

    print('Fingerprinting frozen constants before fitting',flush=True)
    before=constants_digest()
    target_results=[]
    (out/'targets').mkdir(exist_ok=True)
    fit_start=time.perf_counter()
    for index,fact in enumerate(corpus['facts']):
        stem=out/'targets'/fact['id']
        metadata_path=stem.with_suffix('.json')
        if metadata_path.exists():
            meta=json.loads(metadata_path.read_text());arrays=np.load(stem.with_suffix('.npz'))
            target=dict(current=arrays['current'],targets=arrays['targets'],
                key_sets=[{int(k):tuple(v) for k,v in row.items()} for row in meta['key_sets']],
                wall_seconds=meta['wall_seconds'],history=meta['history'])
        else:
            print(f"Fitting {index+1}/{len(corpus['facts'])}: {fact['id']}",flush=True)
            def log(row):
                append(out/'training.jsonl',dict(fact=fact['id'],**row))
                if row['step']%5==0:print(json.dumps(dict(fact=fact['id'],**row)),flush=True)
            target=fit_target(model,memory,tokenizer,fact,steps=args.steps,
                expressions=fact['expressions'],contexts=corpus['protocol']['contexts'],log=log)
            np.savez(stem.with_suffix('.npz'),current=target['current'],targets=target['targets'])
            save(metadata_path,{k:v for k,v in target.items() if k not in ['current','targets','delta']})
        target_results.append(target)
    updates,allocation=allocate_updates(target_results,
        calibration_texts=[d['text'] for d in corpus['facts']+corpus['distractors']],tokenizer=tokenizer)
    memory.add_updates(updates)
    memory.save(out/'memory',dict(config=config,allocation=allocation))
    fit_seconds=time.perf_counter()-fit_start
    save(out/'fitting.json',dict(seconds=fit_seconds,facts=len(corpus['facts']),allocation=allocation,
        per_fact_seconds=[t['wall_seconds'] for t in target_results]))
    print('Fresh fitting finished '+json.dumps(allocation),flush=True)
    correct_residuals=dict(memory.residuals)
    shuffled={}
    rng=np.random.default_rng(91)
    for key,values in correct_residuals.items():
        occupied=list(range(1,values.shape[0]));rng.shuffle(occupied)
        perm=list(range(values.shape[0]))
        for a,b in zip(occupied,occupied[1:]+occupied[:1]):perm[a]=b
        shuffled[key]=values[mx.array(perm)]
    sampler=make_sampler(temp=0)
    print('Loading small BGE encoder on CPU for real online retrieval timing',flush=True)
    embed_id=embedding_receipt['model'];embed_revision=embedding_receipt['revision']
    embed_tok=AutoTokenizer.from_pretrained(embed_id,revision=embed_revision,local_files_only=True)
    embed_model=AutoModel.from_pretrained(embed_id,revision=embed_revision,local_files_only=True).eval().to('cpu')
    torch.set_num_threads(6)
    arrays=np.load(args.data/'embeddings.npz')
    vectors=arrays['documents'].copy()

    def embed(text,query=True):
        if query:text='Represent this sentence for searching relevant passages: '+text
        encoded=embed_tok([text],padding=True,truncation=True,max_length=256,return_tensors='pt')
        with torch.no_grad():
            vector=embed_model(**encoded).last_hidden_state[:,0]
            return torch.nn.functional.normalize(vector,p=2,dim=1).numpy()[0]

    retriever=Retriever(corpus,vectors,{})
    prefix_caches={}
    rows=[]
    fact_by_id={f['id']:f for f in corpus['facts']}
    key_to_facts=defaultdict(set)
    for fact in corpus['facts']:
        for expression in fact['expressions']:
            _,keys=position_and_keys(tokenizer,expression,fact['subject'])
            for length,key in keys.items():key_to_facts[(length,key)].add(fact['id'])

    def activated_facts(tokens):
        found=set()
        for length in (2,3,4):
            for end in range(length,len(tokens)+1):
                key=tuple(tokens[end-length:end])
                if 2 not in key[:-1]:found.update(key_to_facts.get((length,key),()))
        return found

    def evaluate(question,method,style,phase='initial'):
        started=time.perf_counter()
        base_method=method.removesuffix('_cached')
        memory.enabled=base_method in ['memory','shuffled']
        memory.residuals=shuffled if method=='shuffled' else correct_residuals
        documents=[];retrieval_seconds=0.;cache_hit=False
        if base_method in ['oracle','kg','bm25','dense','hybrid']:
            retrieval_start=time.perf_counter()
            if base_method in ['dense','hybrid']:
                retriever.query_vectors[question['id']]=embed(question['text'])
            documents=retriever.retrieve(question,base_method)
            retrieval_seconds=time.perf_counter()-retrieval_start
        if question['kind']=='locality':
            instruction='Answer the general knowledge question briefly and exactly. '
        else:
            instruction=('Use the frozen reference, which can be stored internally or supplied below. '
                'Its fictional or anonymized records override ordinary world knowledge. '
                'If the reference has no matching record, answer UNKNOWN. '
                'This is a record-lookup task, not a treatment claim. ')
        instruction+=('Reason briefly, naming intermediate entities, and finish with Answer: <answer>.\n' if style=='cot'
                      else 'Return only the precise answer label, without explanation.\n')
        context=''
        if base_method in ['oracle','kg','bm25','dense','hybrid']:
            context='Reference records:\n'+('\n'.join(d['text'] for d in documents) if documents else 'No matching records.')+'\n'
        prefix=instruction+context
        suffix='Question: '+question['text']+('\nThoughts:' if style=='cot' else '\nAnswer:')
        prompt=prefix+suffix
        tokens=tokenizer.encode(prompt,add_special_tokens=False)
        prompt_cache=None;remaining=tokens
        if method.endswith('_cached'):
            cut=max(0,len(tokenizer.encode(prefix,add_special_tokens=False))-3)
            prefix_tokens=tokens[:cut]
            cache_key=(phase,base_method,tuple(prefix_tokens))
            if cache_key in prefix_caches:
                cache_hit=True
            else:
                new_cache=model.make_cache()
                logits=model(mx.array([prefix_tokens]),cache=new_cache)
                mx.eval(logits)
                prefix_caches[cache_key]=new_cache
            prompt_cache=copy.deepcopy(prefix_caches[cache_key])
            remaining=tokens[cut:]
        hit_before=dict(memory._hit_counts)
        generation_start=time.perf_counter();pieces=[];generated=[];response=None
        for response in stream_generate(model,tokenizer,remaining,prompt_cache=prompt_cache,
                max_tokens=128 if style=='cot' else 32,sampler=sampler):
            pieces.append(response.text);generated.append(response.token)
        if response is None:raise RuntimeError('No generated answer')
        generation_seconds=time.perf_counter()-generation_start
        text=''.join(pieces);answer=extract_answer(text,cot=style=='cot')
        needed=set(question['needed']);retrieved={d['id'] for d in documents}
        row=dict(question_id=question['id'],suite=question['suite'],component=question['component'],
            kind=question['kind'],style=style,method=method,phase=phase,
            answers=question['answers'],answer=answer,generation=text,
            correct=matches_answer(answer,question['answers']),
            abstained=matches_answer(answer,['UNKNOWN']),
            prompt_tokens=len(tokens),generated_tokens=response.generation_tokens,
            retrieval_seconds=retrieval_seconds,generation_seconds=generation_seconds,
            wall_seconds=time.perf_counter()-started,prefix_cache_hit=cache_hit,
            retrieved_ids=sorted(retrieved),needed_ids=sorted(needed),
            support_coverage=len(needed&retrieved)/len(needed) if needed else None,
            prompt_activated_fact_ids=sorted(activated_facts(tokens)) if memory.enabled else [],
            all_activated_fact_ids=sorted(activated_facts(tokens+generated)) if memory.enabled else [],
            edit_activations={str(k):memory._hit_counts[k]-hit_before[k] for k in hit_before},
            peak_metal_gb=mx.get_peak_memory()/1e9)
        rows.append(row);append(out/'predictions.jsonl',row)
        if len(rows)%20==0:print('Evaluated '+str(len(rows))+' answers',flush=True)
        mx.clear_cache()
        return row

    existing=set()
    if args.resume and (out/'predictions.jsonl').exists():
        rows.extend(json.loads(line) for line in (out/'predictions.jsonl').read_text().splitlines())
        existing={(r['question_id'],r['style'],r['method'],r['phase']) for r in rows}
    randomizer=random.Random(91)
    for question in corpus['questions']:
        styles=['standard','cot'] if question['kind']=='composition' else ['standard']
        methods=['base','memory','shuffled'] if question['kind']=='locality' else list(METHODS)
        for style in styles:
            randomizer.shuffle(methods)
            for method in methods:
                if (question['id'],style,method,'initial') not in existing:evaluate(question,method,style)
        if question['suite']!='locality':
            start=time.perf_counter();records=retriever.kg(question,depth=question.get('hops',1))
            answer=records[-1]['target'] if records else 'UNKNOWN'
            row=dict(question_id=question['id'],suite=question['suite'],component=question['component'],
                kind=question['kind'],style='symbolic',method='kg_only',phase='initial',answer=answer,
                answers=question['answers'],correct=matches_answer(answer,question['answers']),
                wall_seconds=time.perf_counter()-start,prompt_tokens=0,generated_tokens=0,
                needed_ids=question['needed'],retrieved_ids=[d['id'] for d in records],
                retrieval_seconds=time.perf_counter()-start,generation_seconds=0.,peak_metal_gb=0.)
            if (question['id'],'symbolic','kg_only','initial') not in existing:
                rows.append(row);append(out/'predictions.jsonl',row)
    # Sequential corrections are stipulated interventions on the benchmark view,
    # never edits to claims about actual ARAX biology.
    memory.residuals=correct_residuals;memory.enabled=True
    initial_snapshot=memory.snapshot()
    chemical=next(f for f in corpus['facts'] if f['suite']=='arax' and f.get('relation')=='selected_gene')
    gene_facts=[f for f in corpus['facts'] if f['suite']=='arax' and f.get('relation')=='selected_disease']
    saved_corpus=copy.deepcopy(corpus)
    update_reports=[]
    for cycle,gene_fact in enumerate([gene_facts[1],gene_facts[2],gene_facts[0]],1):
        phase=f'update-{cycle}'
        print('Sequential correction '+phase,flush=True)
        update_request={**chemical,'target':gene_fact['subject']}
        start=time.perf_counter()
        fitted=fit_target(model,memory,tokenizer,update_request,steps=args.steps,
            expressions=chemical['expressions'],contexts=corpus['protocol']['contexts'],
            log=lambda row:append(out/'training.jsonl',dict(fact=chemical['id'],phase=phase,**row)))
        delta,report=allocate_updates([fitted],calibration_texts=[d['text'] for d in corpus['facts']],tokenizer=tokenizer)
        memory.add_updates(delta);correct_residuals=dict(memory.residuals)
        edit_seconds=time.perf_counter()-start
        changed=next(f for f in corpus['facts'] if f['id']==chemical['id'])
        changed['target']=gene_fact['subject'];changed['text']=changed['prompt'].format(changed['subject'])+' '+changed['target']+'.'
        for q in corpus['questions']:
            if chemical['id'] in q['needed']:
                q['answers']=[gene_fact['target'] if q['kind']=='composition' else gene_fact['subject']]
                if q['kind']=='composition':q['needed']=[chemical['id'],gene_fact['id']]
        index_start=time.perf_counter()
        doc_index=next(i for i,d in enumerate(retriever.docs) if d['id']==chemical['id'])
        vectors[doc_index]=embed(changed['text'],query=False)
        retriever=Retriever(corpus,vectors,{})
        index_seconds=time.perf_counter()-index_start
        update_reports.append(dict(phase=phase,target=gene_fact['subject'],memory_fit_seconds=edit_seconds,
            retrieval_reindex_seconds=index_seconds,allocation=report))
        prefix_caches.clear()
        for q in corpus['questions']:
            if q['suite']=='arax' and (q['kind']!='composition' or q['id'].endswith('-0')):
                style='cot' if q['kind']=='composition' else 'standard'
                for method in ['memory','kg','hybrid']:evaluate(q,method,style,phase)
        memory.residuals=correct_residuals;memory.enabled=True
        memory.save(out/'memory'/phase,dict(target=gene_fact['subject'],allocation=report))
    rollback_start=time.perf_counter();memory.restore(initial_snapshot)
    rollback_seconds=time.perf_counter()-rollback_start
    correct_residuals=dict(memory.residuals);corpus=saved_corpus
    retriever=Retriever(corpus,arrays['documents'],{})
    prefix_caches.clear()
    for q in corpus['questions']:
        if q['suite']=='arax' and (q['kind']!='composition' or q['id'].endswith('-0')):
            evaluate(q,'memory','cot' if q['kind']=='composition' else 'standard','rollback')
    print('Fingerprinting constants after edits',flush=True)
    memory._probe_delta=None
    after=constants_digest()
    if before!=after:raise RuntimeError('Frozen native constants changed')
    save(out/'summary.json',dict(completed=True,predictions=len(rows),fit_seconds=fit_seconds,
        updates=update_reports,rollback_seconds=rollback_seconds,peak_metal_gb=mx.get_peak_memory()/1e9,
        frozen_constants=dict(before=before,after=after,unchanged=before==after)))
    print('COMPLETED '+json.dumps(dict(predictions=len(rows),peak_metal_gb=mx.get_peak_memory()/1e9)),flush=True)


if __name__=='__main__':main()
