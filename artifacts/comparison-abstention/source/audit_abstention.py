"""Secondary absent-record audit with neutral aliases instead of 'Unknown'."""

import hashlib
import json
from pathlib import Path
import random
import time

from huggingface_hub import snapshot_download
import mlx.core as mx
from mlx_lm import stream_generate
from mlx_lm.sample_utils import make_sampler
from mlx_lm.tokenizer_utils import TokenizerWrapper
from mlx_lm.utils import load_model
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizerFast

from engram_poc.longcat import MODEL_ID, MODEL_REVISION, install_memory
from engram_poc.longcat_keys import extract_answer, matches_answer
from engram_poc.retrieval import Retriever


def main():
    out=Path('runs/comparison-abstention')
    if out.exists():raise ValueError('Preserve the existing audit; choose another output')
    out.mkdir(parents=True)
    root=Path('data/processed/comparison-v1');run=Path('runs/comparison-v1')
    assert json.loads((run/'summary.json').read_text())['completed']
    corpus=json.loads((root/'dataset.json').read_text());receipt=json.loads((root/'embedding-receipt.json').read_text())
    assert hashlib.sha256((root/'dataset.json').read_bytes()).hexdigest()==receipt['corpus_sha256']
    aliases=['Chemical Zera Kavon','Chemical Navo Reltik','Chemical Evar Talmin',
        'Chemical Orla Vensik','Chemical Kira Mantov','Chemical Yavo Delsin']
    questions=[dict(id=f'neutral-absent-{i}',suite='arax',kind='unknown',entity=alias,needed=[],answers=['UNKNOWN'],
        text=f'What selected gene neighbor does the frozen snapshot view record for {alias}?') for i,alias in enumerate(aliases)]
    methods=['base','memory','kg','bm25','dense','hybrid']
    config=dict(model=MODEL_ID,revision=MODEL_REVISION,methods=methods,questions=questions,
        dataset_sha256=receipt['corpus_sha256'],
        status='Secondary audit added after identifying the literal Unknown cue in initial absent-entity aliases',
        memory_sha256=json.loads((run/'memory/manifest.json').read_text())['updates_sha256'],
        max_tokens=32,seed=841)
    (out/'source').mkdir();code=hashlib.sha256()
    for source in [Path(__file__),*sorted(Path('engram_poc').glob('*.py'))]:
        code.update(source.name.encode());code.update(source.read_bytes());(out/'source'/source.name).write_bytes(source.read_bytes())
    config['source_sha256']=code.hexdigest();(out/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    mx.set_memory_limit(48_000_000_000);mx.set_cache_limit(128*1024**2)
    path=snapshot_download(MODEL_ID,revision=MODEL_REVISION,local_files_only=True,allow_patterns=['*.json','*.jinja','*.safetensors'])
    model,_=load_model(Path(path));memory=install_memory(model,run/'memory')
    tokenizer=TokenizerWrapper(PreTrainedTokenizerFast.from_pretrained(path,local_files_only=True),eos_token_ids=[2])
    torch.set_num_threads(6)
    embed_tok=AutoTokenizer.from_pretrained(receipt['model'],revision=receipt['revision'],local_files_only=True)
    embed_model=AutoModel.from_pretrained(receipt['model'],revision=receipt['revision'],local_files_only=True).eval().to('cpu')
    retriever=Retriever(corpus,np.load(root/'embeddings.npz')['documents'],{})
    sampler=make_sampler(temp=0);rows=[];rng=random.Random(841)
    instruction=('Use the frozen reference, which can be stored internally or supplied below. '
        'Its fictional or anonymized records override ordinary world knowledge. '
        'If the reference has no matching record, answer UNKNOWN. '
        'This is a record-lookup task, not a treatment claim. '
        'Return only the precise answer label, without explanation.\n')
    for question in questions:
        order=list(methods);rng.shuffle(order)
        for method in order:
            memory.enabled=method=='memory';start=time.perf_counter()
            if method in ['dense','hybrid']:
                encoded=embed_tok(['Represent this sentence for searching relevant passages: '+question['text']],return_tensors='pt',truncation=True,max_length=256)
                with torch.no_grad():
                    vector=embed_model(**encoded).last_hidden_state[:,0]
                    retriever.query_vectors[question['id']]=torch.nn.functional.normalize(vector,p=2,dim=1).numpy()[0]
            docs=retriever.retrieve(question,method) if method in ['kg','bm25','dense','hybrid'] else []
            context=('Reference records:\n'+('\n'.join(d['text'] for d in docs) if docs else 'No matching records.')+'\n') if method in ['kg','bm25','dense','hybrid'] else ''
            prompt=instruction+context+'Question: '+question['text']+'\nAnswer:'
            hits=dict(memory._hit_counts);pieces=[]
            for response in stream_generate(model,tokenizer,tokenizer.encode(prompt,add_special_tokens=False),sampler=sampler,max_tokens=32):pieces.append(response.text)
            text=''.join(pieces);answer=extract_answer(text)
            row=dict(question_id=question['id'],question=question['text'],method=method,generation=text,answer=answer,
                correct=matches_answer(answer,['UNKNOWN']),retrieved_ids=[d['id'] for d in docs],
                wall_seconds=time.perf_counter()-start,generated_tokens=response.generation_tokens,
                edit_activations={str(k):memory._hit_counts[k]-hits[k] for k in hits})
            rows.append(row)
            with (out/'predictions.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
            print(json.dumps(row),flush=True);mx.clear_cache()
    summary=dict(completed=True,rows=len(rows),peak_metal_gb=mx.get_peak_memory()/1e9)
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print('COMPLETED',flush=True)


if __name__=='__main__':main()
