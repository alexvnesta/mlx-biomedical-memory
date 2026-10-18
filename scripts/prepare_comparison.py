"""Freeze tasks and precompute a genuine BGE dense-retrieval baseline on CPU."""

import argparse
import hashlib
import json
from pathlib import Path
import time
import shutil

import truststore
truststore.inject_into_ssl()

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from engram_poc.comparison_data import build


root = Path('data/processed/comparison-v1')
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--replay',action='store_true',help='Use the exact public corpus/vectors, and cache the pinned CPU encoder; no live ARAX snapshots needed')
args=parser.parse_args()
if args.replay:
    source=Path('artifacts/comparison-v1');root.mkdir(parents=True,exist_ok=True)
    for name in ['dataset.json','embeddings.npz','embedding-receipt.json']:
        shutil.copy2(source/name,root/name)
    receipt=json.loads((root/'embedding-receipt.json').read_text())
    assert hashlib.sha256((root/'dataset.json').read_bytes()).hexdigest()==receipt['corpus_sha256']
    assert hashlib.sha256((root/'embeddings.npz').read_bytes()).hexdigest()==receipt['embeddings_sha256']
    corpus=json.loads((root/'dataset.json').read_text())
else:
    corpus = build(root)
model_id = 'BAAI/bge-small-en-v1.5'
revision = '5c38ec7c405ec4b44b94cc5a9bb96e735b38267a'
print('Loading CPU embedding model', flush=True)
tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
model = AutoModel.from_pretrained(model_id, revision=revision).eval().to('cpu')
torch.set_num_threads(6)
if args.replay:
    assert receipt['model']==model_id and receipt['revision']==revision
    print('Verified exact public corpus/vectors and cached the pinned CPU encoder.',flush=True)
    raise SystemExit(0)

def encode(texts):
    vectors = []
    start = time.perf_counter()
    with torch.no_grad():
        for offset in range(0, len(texts), 32):
            encoded = tokenizer(texts[offset:offset+32], padding=True, truncation=True, max_length=256, return_tensors='pt')
            hidden = model(**encoded).last_hidden_state[:, 0]
            vectors.append(torch.nn.functional.normalize(hidden, p=2, dim=1).numpy())
    return np.concatenate(vectors), time.perf_counter()-start

documents = corpus['facts']+corpus['distractors']
doc_vectors, doc_seconds = encode([d['text'] for d in documents])
query_vectors, query_seconds = encode(['Represent this sentence for searching relevant passages: '+q['text'] for q in corpus['questions']])
np.savez(root/'embeddings.npz', documents=doc_vectors, queries=query_vectors)
receipt = dict(model=model_id, revision=revision, pooling='CLS then L2 normalization',
    device='CPU', dimension=int(doc_vectors.shape[1]), documents=len(documents), queries=len(corpus['questions']),
    corpus_sha256=hashlib.sha256((root/'dataset.json').read_bytes()).hexdigest(),
    embeddings_sha256=hashlib.sha256((root/'embeddings.npz').read_bytes()).hexdigest(),
    document_embedding_seconds=doc_seconds, query_embedding_seconds=query_seconds,
    query_seconds_per_item=query_seconds/len(corpus['questions']),
    caveat='Query vectors are batched preparation diagnostics only. The comparison runners perform and time actual online CPU query encodings; these amortized preparation times are not used as online latency.')
(root/'embedding-receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
print(json.dumps(receipt, indent=2), flush=True)
