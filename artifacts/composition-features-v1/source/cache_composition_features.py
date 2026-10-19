"""Cache causal training/dev features with matched noisy atomic prefixes."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import mlx.core as mx

from engram_poc.decisive_model import load_reader,SPLIT_LAYER
from engram_poc.decisive_prompts import encode_prompt,target_text
from engram_poc.decisive_data import fields,atomic_solution
from engram_poc.readout_data import noisy_prefix
from engram_poc.longcat import MODEL_ID,MODEL_REVISION
from engram_poc.longcat_runtime import snapshot_source,save,append
from engram_poc.resource_guard import ResourceGuard
from cache_decisive_features import digest_frozen


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=Path('data/processed/composition-v1/dataset.json'))
    p.add_argument('--output',type=Path,default=Path('runs/composition-features-v1'))
    p.add_argument('--resume',action='store_true')
    args=p.parse_args();out=args.output
    if out.exists() and not args.resume:raise RuntimeError('Preserve previous features')
    out.mkdir(exist_ok=True);(out/'features').mkdir(exist_ok=True)
    corpus=json.loads(args.data.read_text());data_sha=hashlib.sha256(args.data.read_bytes()).hexdigest()
    if not args.resume:
        save(out/'config.json',dict(model=MODEL_ID,revision=MODEL_REVISION,dataset_sha256=data_sha,
            source_sha256=snapshot_source(__file__,out/'source'),split_layer=SPLIT_LAYER,
            atomic_prefix_noise=.5,feature_semantics='Causal train/dev only; true labels; matched noisy past atomic values',
            dependency_sha256=hashlib.sha256(Path('scripts/cache_decisive_features.py').read_bytes()).hexdigest()))
        (out/'dataset.json').write_bytes(args.data.read_bytes())
        (out/'cache_decisive_features.py').write_bytes(Path('scripts/cache_decisive_features.py').read_bytes())
        (out/'prepare_composition_experiment.py').write_bytes(Path('scripts/prepare_composition_experiment.py').read_bytes())
    else:
        cfg=json.loads((out/'config.json').read_text());assert cfg['dataset_sha256']==data_sha
        h=hashlib.sha256()
        for path in [Path(__file__),*sorted(Path('engram_poc').glob('*.py'))]:h.update(path.name.encode());h.update(path.read_bytes())
        assert h.hexdigest()==cfg['source_sha256']
        assert hashlib.sha256(Path('scripts/cache_decisive_features.py').read_bytes()).hexdigest()==cfg['dependency_sha256']
    guard=ResourceGuard();model,tok=load_reader('memory');before=digest_frozen(model)
    rows=[json.loads(x) for x in (out/'manifest.jsonl').read_text().splitlines()] if (out/'manifest.jsonl').exists() else []
    done={(r['id'],r['mode']) for r in rows};parity=[];started=time.perf_counter()
    for split in ['dev','train']:
        for case in corpus['splits'][split]:
            target=tok.encode(target_text(case),add_special_tokens=False)+[2]
            input_target=target[:-1]
            if split=='train' and case['kind']=='atomic':
                seed=int(hashlib.sha256(case['id'].encode()).hexdigest()[:16],16)
                _,input_target=noisy_prefix(atomic_solution(case),lambda s:tok.encode(s,add_special_tokens=False),seed,.5)
            assert len(input_target)+1==len(target)
            for mode in ['text','memory']:
                if (case['id'],mode) in done:continue
                e=encode_prompt(tok,case,mode);tokens=e['ids']+input_target
                h=model.prefix(mx.array([tokens]));mx.eval(h)
                if len(parity)<2:
                    f=mx.array(fields(case['records']));model.begin(f,e['placements'])
                    direct=model(mx.array([tokens]))[:,len(e['ids'])-1:]
                    cached=model.tail(h,f,e['placements'],loss_start=len(e['ids'])-1)
                    error=mx.max(mx.abs(direct-cached)).item();assert error==0
                    parity.append(dict(mode=mode,case_id=case['id'],max_error=error));del direct,cached
                path=out/'features'/(case['id']+'-'+mode+'.safetensors');mx.save_safetensors(str(path),dict(hidden=h))
                row=dict(id=case['id'],split=split,kind=case['kind'],mode=mode,path=str(path),shape=list(h.shape),
                    feature_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),token_sha256=hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),
                    target=target,input_target=input_target,loss_start=len(e['ids'])-1,positions=e['placements'],fields=fields(case['records']))
                append(out/'manifest.jsonl',row);rows.append(row);del h;mx.clear_cache()
                if len(rows)%64==0:print(json.dumps(dict(cached=len(rows),resource=guard.check())),flush=True)
    after=digest_frozen(model);assert before==after
    save(out/'summary.json',dict(completed=True,features=len(rows),seconds=time.perf_counter()-started,parity=parity,
        frozen_constants=dict(before=before,after=after),resource=guard.check()))


if __name__=='__main__':main()
