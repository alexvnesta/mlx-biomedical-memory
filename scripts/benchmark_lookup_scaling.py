"""CPU index microbenchmark: address lookup is not learned-memory quality."""

import argparse
import json
import os
from pathlib import Path
import sqlite3
import statistics
import time

import numpy as np


def summarize(times):
    return dict(p50_ms=1000*statistics.median(times), p95_ms=1000*float(np.percentile(times,95)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('runs/lookup-scaling.json'))
    args = parser.parse_args()
    rng = np.random.default_rng(41)
    rows = []
    for size in [100,1000,10000,100000]:
        build_start = time.perf_counter()
        keys = {(i, i+1, i+2, i+3): i for i in range(size)}
        key_seconds = time.perf_counter()-build_start
        build_start = time.perf_counter()
        graph = {f'Entity{i}':f'Entity{i+1}' for i in range(size)}
        graph_seconds = time.perf_counter()-build_start
        db = sqlite3.connect(':memory:')
        build_start = time.perf_counter()
        db.execute('CREATE VIRTUAL TABLE docs USING fts5(text)')
        db.executemany('INSERT INTO docs(text) VALUES (?)',
            ((f'Entity{i} has recorded neighbor Entity{i+1}.',) for i in range(size)))
        db.commit()
        fts_seconds = time.perf_counter()-build_start
        sqlite_bytes = db.execute('PRAGMA page_size').fetchone()[0]*db.execute('PRAGMA page_count').fetchone()[0]
        build_start = time.perf_counter()
        vectors = rng.normal(size=(size,384)).astype(np.float32)
        vectors /= np.linalg.norm(vectors,axis=1,keepdims=True)
        vector_seconds = time.perf_counter()-build_start
        queries = rng.integers(0,max(size-3,1),100)
        timings = {m:[] for m in ['exact_4gram_address','kg_2hop','sqlite_fts5_top4','dense_cosine_top4']}
        # Dense query encoding is excluded deliberately and prominently. The
        # end-to-end LongCat comparison separately includes actual BGE encoding.
        for i in queries:
            i=int(i); key=(i,i+1,i+2,i+3); entity=f'Entity{i}'; vector=vectors[i]
            t=time.perf_counter(); result=keys.get(key); timings['exact_4gram_address'].append(time.perf_counter()-t)
            assert result==i
            t=time.perf_counter(); result=graph.get(graph.get(entity,''));timings['kg_2hop'].append(time.perf_counter()-t)
            assert result==f'Entity{i+2}'
            t=time.perf_counter()
            result=db.execute('SELECT rowid FROM docs WHERE docs MATCH ? ORDER BY bm25(docs) LIMIT 4',(entity,)).fetchall()
            timings['sqlite_fts5_top4'].append(time.perf_counter()-t)
            assert result
            t=time.perf_counter(); scores=vectors@vector; indices=np.argpartition(scores,-4)[-4:]
            timings['dense_cosine_top4'].append(time.perf_counter()-t)
            assert i in indices
        rows.append(dict(records=size, queries=len(queries), timings={k:summarize(v) for k,v in timings.items()},
            build_seconds=dict(exact_key=key_seconds,kg=graph_seconds,sqlite_fts=fts_seconds,
                random_dense_vectors_not_encoder=vector_seconds),
            storage=dict(sqlite_database_bytes=sqlite_bytes,dense_float32_bytes=vectors.nbytes,
                illustrative_bf16_3072dim_one_residual_per_record_bytes=size*3072*2)))
        db.close(); print(json.dumps(rows[-1]),flush=True)
    output=dict(completed=True,rows=rows,numpy=np.__version__,
        thread_environment={k:os.environ.get(k) for k in ['OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','OMP_NUM_THREADS']},
        limitations=['Synthetic CPU lookup only; no LongCat inference or learned quality scaling measured',
            'Exact address benchmark omits CPU/GPU sync, 2/3/4-gram scans, vector fetch and residual additions',
            'KG receives normalized entity and fixed two-hop plan; parsing is excluded',
            'Dense is exhaustive 384-D cosine, not ANN; query encoding excluded',
            'Dense build uses random vectors, not corpus embedding; not comparable encoder cost',
            'Illustrative residual storage assumes one key per record; real multi-expression edits use more keys',
            'Finite observations do not establish asymptotic guarantees or biological scale performance'])
    args.output.write_text(json.dumps(output,indent=2)+'\n')


if __name__=='__main__':main()
