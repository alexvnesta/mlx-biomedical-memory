"""Evaluate richer relations with explicit proof grading and model-chosen tools."""

import argparse
import copy
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import random
import time

import mlx.core as mx
from mlx_lm import stream_generate
from mlx_lm.sample_utils import make_sampler
import numpy as np
import torch
from transformers import AutoModel,AutoTokenizer

from engram_poc.longcat import MODEL_ID,MODEL_REVISION
from engram_poc.longcat_editor import log_probs
from engram_poc.longcat_keys import extract_answer,matches_answer,normalize_answer
from engram_poc.longcat_runtime import load_native,frozen_digest,snapshot_source,save,append
from engram_poc.scoped_memory import install_scoped
from engram_poc.relational import EvidenceGraph,parse_response,grade,record_text
from engram_poc.relational_prompts import direct_prompt,agent_prompt,subject_position


METHODS=['gold_text','gold_scoped','gold_ngram','gold_shuffled','gold_native',
    'bm25_text','hybrid_text','agent_text','agent_scoped','agent_native']
EMBED_ID='BAAI/bge-small-en-v1.5'
EMBED_REV='5c38ec7c405ec4b44b94cc5a9bb96e735b38267a'


def load_checkpoint(memory,directory):
    out=Path(directory);manifest=json.loads((out/'manifest.json').read_text())
    assert manifest['model_revision']==MODEL_REVISION
    assert hashlib.sha256((out/'scoped.safetensors').read_bytes()).hexdigest()==manifest['vectors_sha256']
    vectors=mx.load(str(out/'scoped.safetensors'))['vectors']
    memory.store(manifest['keys'],[vectors[i] for i in range(1,len(manifest['keys'])+1)],manifest['permutation'][1:])
    ng=json.loads((out/'ngram/manifest.json').read_text())
    assert hashlib.sha256((out/'ngram/updates.safetensors').read_bytes()).hexdigest()==ng['updates_sha256']
    arrays=mx.load(str(out/'ngram/updates.safetensors'))
    memory.add_updates({(int(n),tuple(key)):arrays[n][i+1] for n,keys in ng['keys'].items() for i,key in enumerate(keys)})
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fit',type=Path,default=Path('runs/relational-fit-v1'))
    parser.add_argument('--output',type=Path,default=Path('runs/relational-v1'))
    parser.add_argument('--dev-only',action='store_true')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();out=args.output
    fit_summary=json.loads((args.fit/'summary.json').read_text());assert fit_summary['completed']
    corpus=json.loads((args.fit/'dataset.json').read_text())
    if out.exists() and not args.resume:raise ValueError('Choose a new output or resume the identical recipe')
    out.mkdir(parents=True,exist_ok=True)
    if args.resume:
        old=json.loads((out/'config.json').read_text())
        if (out/'summary.json').exists():raise ValueError('Run already complete')
        code=hashlib.sha256()
        for path in [Path(__file__),*sorted(Path('engram_poc').glob('*.py'))]:
            code.update(path.name.encode());code.update(path.read_bytes())
        assert code.hexdigest()==old['source_sha256'],'Resume source changed'
        config=old
    else:
        config=dict(model=MODEL_ID,revision=MODEL_REVISION,started_utc=datetime.now(timezone.utc).isoformat(),
            dataset_sha256=hashlib.sha256((args.fit/'dataset.json').read_bytes()).hexdigest(),
            checkpoint_sha256=hashlib.sha256((args.fit/'memory/scoped.safetensors').read_bytes()).hexdigest(),
            source_sha256=snapshot_source(__file__,out/'source'),methods=METHODS,seed=811,
            max_tokens=192,dev_only=args.dev_only,protocol=corpus['protocol'],
            reasoning_format='Pinned native chat template, enable_thinking=False; literal recall uses raw continuation')
        save(out/'config.json',config);(out/'dataset.json').write_bytes((args.fit/'dataset.json').read_bytes())
    assert config['dataset_sha256']==hashlib.sha256((args.fit/'dataset.json').read_bytes()).hexdigest()
    assert config['checkpoint_sha256']==hashlib.sha256((args.fit/'memory/scoped.safetensors').read_bytes()).hexdigest()
    print('Loading pinned native model',flush=True);model,tokenizer=load_native();memory=install_scoped(model)
    manifest=load_checkpoint(memory,args.fit/'memory');sampler=make_sampler(temp=0)
    print('Fingerprinting frozen model',flush=True);before=frozen_digest(model)
    graph=EvidenceGraph(corpus['records']);by_id=graph.by_id;motifs={r['id']:r for r in corpus['motifs']}
    rows=[]
    if args.resume and (out/'predictions.jsonl').exists():
        rows=[json.loads(line) for line in (out/'predictions.jsonl').read_text().splitlines()]
    existing={(r['stage'],r['case_id'],r['method'],r.get('form','')) for r in rows}
    rng=random.Random(811)

    def generate(prompt,mode,request,*,json_output=True,max_tokens=192,force_length=False,chat=False):
        if chat:
            prompt=tokenizer.apply_chat_template([dict(role='user',content=prompt)],tokenize=False,
                add_generation_prompt=True,enable_thinking=False)
        memory.begin(mode,[(request['key'],subject_position(tokenizer,prompt,request))])
        tokens=tokenizer.encode(prompt,add_special_tokens=False);pieces=[];obj=None;response=None
        start=time.perf_counter();first=None
        processors=[lambda tokens,logits:logits.at[...,2].add(-1e9)] if force_length else None
        for response in stream_generate(model,tokenizer,tokens,sampler=sampler,max_tokens=max_tokens,logits_processors=processors):
            if first is None:first=time.perf_counter()
            pieces.append(response.text)
            if json_output:
                obj=parse_response(''.join(pieces))
                if obj:break
        if response is None:raise RuntimeError('No response')
        elapsed=time.perf_counter()-start;text=''.join(pieces)
        result=dict(prompt=prompt,generation=text,parsed=obj,prompt_tokens=len(tokens),generated_tokens=response.generation_tokens,
            seconds=elapsed,time_to_first_token_seconds=first-start,mode=mode)
        if force_length:assert response.generation_tokens==max_tokens
        mx.clear_cache();return result

    def record(row):
        rows.append(row);append(out/'predictions.jsonl',row)
        if len(rows)%20==0:print('Completed '+str(len(rows))+' outcomes',flush=True)

    # Development agent probes use disjoint worlds and are never pooled with test.
    def evaluate(case,method,stage='initial',current_graph=None):
        g=current_graph or graph;request=motifs[case['motif_id']]
        textual=method.endswith('_text');mode='scoped' if method.endswith('_scoped') else 'ngram' if method.endswith('_ngram') else 'shuffled' if method.endswith('_shuffled') else 'native'
        start=time.perf_counter();calls=[];retrieved=[];certificates=[];tool_calls=0;tool_errors=0;obj=None
        if method.startswith('agent_'):
            history='Initial state: the graph has not been inspected. Query it before declaring UNKNOWN or NONE.\n'
            for round_index in range(4):
                state=f'\nResponses remaining including this one: {4-round_index}. Lookups remaining: {6-tool_calls}. '
                if round_index==3:state+='This is the last response: return the final answer and evidence JSON.'
                response=generate(agent_prompt(case,request,textual,history+state),mode,request,chat=True)
                calls.append(response);obj=response['parsed']
                if obj and 'answer' in obj:break
                proposals=obj.get('lookups',[]) if obj else []
                if not isinstance(proposals,list):proposals=[]
                results=[]
                for lookup in proposals[:2]:
                    if tool_calls>=6:break
                    tool_calls+=1
                    try:
                        if not isinstance(lookup,dict) or not isinstance(lookup.get('entity'),str):raise ValueError('entity must be a string')
                        context=lookup.get('context') or {}
                        if not isinstance(context,dict) or not all(isinstance(k,str) and isinstance(v,str) for k,v in context.items()):raise ValueError('context must map strings to strings')
                        result=g.lookup(lookup['entity'],lookup.get('relation'),context,lookup.get('direction','both'))
                        retrieved.extend(r['id'] for r in result['records'])
                        certificates.append(dict(entity=lookup['entity'],relation=lookup.get('relation'),context=context,
                            direction=lookup.get('direction','both'),complete=result['complete'],record_ids=[r['id'] for r in result['records']]))
                        results.append(dict(request=lookup,complete=result['complete'],total_matching=result['total_matching'],
                            records=[record_text(r) for r in result['records']]))
                    except (ValueError,TypeError) as error:
                        tool_errors+=1;results.append(dict(error=str(error)))
                history+='\nAgent JSON: '+response['generation']+'\nTool results: '+json.dumps(results)+'\n'
                if not results:history+='Return a valid lookup or a final answer JSON.\n'
            obj=obj if obj and 'answer' in obj else {}
        else:
            if method.startswith('gold_'):documents=[g.by_id[i] for i in case['record_ids']]
            elif method=='bm25_text':documents=g.search(case['question'],limit=12)
            elif method=='hybrid_text':documents=hybrid(case['question'])
            else:raise ValueError(method)
            retrieved=[r['id'] for r in documents]
            if method.startswith('gold_'):
                # Oracle evidence is explicitly the complete local world. Its
                # completeness is a grader input, never a hidden agent plan.
                certificates=[dict(entity=entity,relation=None,context={},direction='both',complete=True)
                    for entity in {x for r in documents for x in [r['subject'],r['object']]}]
            response=generate(direct_prompt(case,request,documents,textual),mode,request,chat=True)
            calls.append(response);obj=response['parsed'] or {}
        evidence=obj.get('evidence',[])
        if not isinstance(evidence,list) or not all(isinstance(x,str) for x in evidence):evidence=[]
        answer=str(obj.get('answer','INVALID')).strip().upper()
        row=dict(stage=stage,case_id=case['id'],suite=case['suite'],family=case['family'],method=method,
            expected=case['answer'],answer=answer,evidence=evidence,retrieved_ids=sorted(set(retrieved)),
            completeness_certificates=certificates,
            tool_calls=tool_calls,tool_errors=tool_errors,model_calls=len(calls),calls=calls,
            prompt_tokens=sum(r['prompt_tokens'] for r in calls),generated_tokens=sum(r['generated_tokens'] for r in calls),
            wall_seconds=time.perf_counter()-start,**grade(case,answer,evidence,retrieved,
                certificates=certificates,records=[g.by_id[i] for i in set(retrieved)]))
        record(row)

    if args.dev_only:
        for case in corpus['dev_cases']:
            for method in ['gold_text','agent_text']:evaluate(case,method,'development')
        save(out/'summary.json',dict(completed=True,rows=len(rows),development_only=True));return

    # Embedding preparation and actual online query encoding are CPU-only.
    print('Preparing the pinned BGE hybrid index on CPU',flush=True);torch.set_num_threads(6)
    embed_tok=AutoTokenizer.from_pretrained(EMBED_ID,revision=EMBED_REV,local_files_only=True)
    embed_model=AutoModel.from_pretrained(EMBED_ID,revision=EMBED_REV,local_files_only=True).eval().to('cpu')
    def encode(texts,query=False):
        if query:texts=['Represent this sentence for searching relevant passages: '+text for text in texts]
        batches=[]
        with torch.no_grad():
            for offset in range(0,len(texts),32):
                encoded=embed_tok(texts[offset:offset+32],padding=True,truncation=True,max_length=256,return_tensors='pt')
                vector=embed_model(**encoded).last_hidden_state[:,0]
                batches.append(torch.nn.functional.normalize(vector,p=2,dim=1).numpy())
        return np.concatenate(batches)
    t=time.perf_counter();vectors=encode([record_text(r) for r in corpus['records']]);index_seconds=time.perf_counter()-t
    np.savez(out/'document-embeddings.npz',vectors=vectors)
    save(out/'embedding-receipt.json',dict(model=EMBED_ID,revision=EMBED_REV,device='CPU',pooling='CLS+L2',
        documents=len(vectors),document_encoding_seconds=index_seconds,
        embeddings_sha256=hashlib.sha256((out/'document-embeddings.npz').read_bytes()).hexdigest()))
    def hybrid(question):
        query=encode([question],True)[0];dense=np.argsort(-(vectors@query));ranks=np.zeros(len(vectors))
        positions={r['id']:i for i,r in enumerate(corpus['records'])}
        for rank,index in enumerate(dense):ranks[index]+=1/(61+rank)
        for rank,r in enumerate(graph.search(question,limit=len(vectors))):ranks[positions[r['id']]]+=1/(61+rank)
        return [corpus['records'][i] for i in np.argsort(-ranks)[:12]]

    requests=corpus['atomic_requests']+corpus['motifs']
    for request in requests:
        forms=['canonical','question'] if request['id'].startswith('atomic-') else ['canonical']
        for form in forms:
            conditions=['native','ngram','scoped','shuffled','text'];rng.shuffle(conditions)
            for method in conditions:
                key=('recall',request['id'],method,form)
                if key in existing:continue
                cue=request['prompt'].format(request['subject'])
                prompt='Return only the literal stored value, without explanation.\n'+cue
                if method=='text':prompt+=' '+request['target']+('\nRepeat the stored value:' if form=='canonical' else '\nQuestion: What value does this exact scoped lookup record?\nAnswer:')
                elif form=='question':prompt+='\nWhat is the value of this exact scoped lookup?\nAnswer:'
                response=generate(prompt,'native' if method=='text' else method,request,json_output=False,max_tokens=128)
                answer=extract_answer(response['generation']);v,t=normalize_answer(answer),normalize_answer(request['target'])
                prefix=v.startswith(t) and (len(v)==len(t) or not(v[len(t)].isalnum() or v[len(t)]=='_'))
                memory.begin('native' if method=='text' else method,[(request['key'],subject_position(tokenizer,prompt,request))])
                prompt_ids=tokenizer.encode(prompt,add_special_tokens=False)
                target_ids=tokenizer.encode(' '+request['target'].lstrip(),add_special_tokens=False)
                lp=log_probs(model(mx.array([prompt_ids+target_ids[:-1]]))[0,len(prompt_ids)-1:])
                nll=-mx.mean(mx.take_along_axis(lp,mx.array(target_ids)[:,None],axis=-1));mx.eval(nll)
                row=dict(stage='recall',case_id=request['id'],suite='atomic' if request['id'].startswith('atomic-') else 'motif',
                    family=request.get('family','tuple'),method=method,form=form,answer=answer,expected=request['target'],
                    correct=matches_answer(answer,[request['target']]),target_prefix_correct=prefix,
                    teacher_forced_nll=nll.item(),
                    wall_seconds=response['seconds'],calls=[response],model_calls=1)
                record(row)
    for case in corpus['cases']:
        methods=list(METHODS);rng.shuffle(methods)
        for method in methods:
            if ('initial',case['id'],method,'') not in existing:evaluate(case,method)
    for intervention in corpus['interventions']:
        case=intervention['case'];replace={r['id']:r for r in intervention['replacement_records']}
        updated=EvidenceGraph([replace.get(r['id'],r) for r in corpus['records']])
        for method in ['gold_text','gold_scoped','agent_text','agent_scoped','agent_native']:
            if ('intervention',case['id'],method,'') not in existing:evaluate(case,method,'intervention',updated)
    # Fixed-length throughput is secondary and cannot establish equal-quality speed.
    latency_cases=random.Random(912).sample(corpus['cases'][:48],12)
    for repeat in range(2):
        for case in latency_cases:
            request=motifs[case['motif_id']];methods=['gold_text','gold_scoped','gold_ngram'];rng.shuffle(methods)
            for method in methods:
                identifier=case['id']+'-repeat-'+str(repeat)
                if ('latency',identifier,method,'') in existing:continue
                started=time.perf_counter()
                prompt=direct_prompt(case,request,[by_id[i] for i in case['record_ids']],method=='gold_text')
                response=generate(prompt,'scoped' if method=='gold_scoped' else 'ngram' if method=='gold_ngram' else 'native',request,
                    json_output=False,max_tokens=32,force_length=True,chat=True)
                record(dict(stage='latency',case_id=identifier,suite=case['suite'],family=case['family'],method=method,
                    calls=[response],wall_seconds=time.perf_counter()-started,model_calls=1,generated_tokens=response['generated_tokens'],
                    prompt_tokens=response['prompt_tokens']))
    print('Fingerprinting frozen model after inference',flush=True);memory._probe_delta=None;after=frozen_digest(model)
    assert before==after
    save(out/'summary.json',dict(completed=True,rows=len(rows),peak_metal_gb=mx.get_peak_memory()/1e9,
        frozen_constants=dict(before=before,after=after,unchanged=before==after)))
    print('COMPLETED '+str(len(rows))+' outcomes',flush=True)


if __name__=='__main__':main()
