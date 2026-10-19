"""Predict path signs from fixed input-only prefixes and changed record values."""
import argparse
import hashlib
import json
from pathlib import Path

import mlx.core as mx
from engram_poc.decisive_model import load_reader
from engram_poc.longcat_runtime import save,append,snapshot_source
from engram_poc.resource_guard import ResourceGuard
from cache_decisive_features import digest_frozen


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--training',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--features',type=Path,default=Path('runs/composition-oracle-features-v1'))
    p.add_argument('--resume',action='store_true');args=p.parse_args();out=args.output
    trained=json.loads((args.training/'config.json').read_text());mode=trained['mode'];checkpoint=args.training/'selected.safetensors'
    if not args.resume:
        out.mkdir(exist_ok=False)
        save(out/'config.json',dict(source_sha256=snapshot_source(__file__,out/'source'),mode=mode,seed=trained['seed'],
            training=str(args.training),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            manifest_sha256=hashlib.sha256((args.features/'manifest.jsonl').read_bytes()).hexdigest(),
            dependency_sha256=hashlib.sha256(Path('scripts/cache_decisive_features.py').read_bytes()).hexdigest(),
            semantics='Next sign token, with oracle path input; no free-path credit'))
        (out/'cache_decisive_features.py').write_bytes(Path('scripts/cache_decisive_features.py').read_bytes())
    else:
        cfg=json.loads((out/'config.json').read_text())
        assert cfg['checkpoint_sha256']==hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        assert cfg['manifest_sha256']==hashlib.sha256((args.features/'manifest.jsonl').read_bytes()).hexdigest()
        source=hashlib.sha256()
        for path in [Path(__file__),*sorted(Path('engram_poc').glob('*.py'))]:source.update(path.name.encode());source.update(path.read_bytes())
        assert source.hexdigest()==cfg['source_sha256']
        assert hashlib.sha256(Path('scripts/cache_decisive_features.py').read_bytes()).hexdigest()==cfg['dependency_sha256']
    assert json.loads((args.features/'summary.json').read_text())['completed']
    signids={k:v[0] for k,v in json.loads((args.features/'sign-token-ids.json').read_text()).items()}
    guard=ResourceGuard();model,tok=load_reader(mode,tail_only=True);model.load_weights(str(checkpoint),strict=False);mx.eval(model.parameters())
    before=digest_frozen(model)
    manifest=[json.loads(x) for x in (args.features/'manifest.jsonl').read_text().splitlines()]
    rows=[json.loads(x) for x in (out/'predictions.jsonl').read_text().splitlines()] if (out/'predictions.jsonl').exists() else []
    done={(r['case_id'],r['condition']) for r in rows}
    for r in manifest:
        if r['mode']!=mode:continue
        for condition in (['normal','masked'] if mode=='memory' else ['normal']):
            if (r['case_id'],condition) in done:continue
            path=Path(r['path']);assert hashlib.sha256(path.read_bytes()).hexdigest()==r['feature_sha256']
            h=mx.load(str(path))['hidden'];f=mx.array(r['fields']);model.begin(f,r['positions'],enabled=condition!='masked')
            logits=model.tail(h,f,r['positions'],loss_start=h.shape[1]-1)[0,0].astype(mx.float32)
            token=mx.argmax(logits).item()
            row=dict(case_id=r['case_id'],world_id=r['world_id'],depth=r['depth'],nuisance=r['nuisance'],condition=condition,
                expected_sign=r['expected_sign'],prediction_token=token,prediction_text=tok.decode([token]),
                correct=token==signids[r['expected_sign']],logits={k:logits[v].item() for k,v in signids.items()})
            append(out/'predictions.jsonl',row);rows.append(row);del h,logits;mx.clear_cache()
            if len(rows)%64==0:print(json.dumps(dict(evaluated=len(rows),resource=guard.check())),flush=True)
    after=digest_frozen(model);assert before==after
    results={}
    for condition in sorted({r['condition'] for r in rows}):
        results[condition]={}
        for depth in [1,2,3,4]:
            subset=[r for r in rows if r['condition']==condition and r['depth']==depth]
            worlds={r['world_id'] for r in subset}
            results[condition][str(depth)]=dict(correct=sum(r['correct'] for r in subset),total=len(subset),
                worlds_all_correct=sum(all(r['correct'] for r in subset if r['world_id']==w) for w in worlds),worlds=len(worlds))
    save(out/'summary.json',dict(completed=True,rows=len(rows),results=results,
        frozen_constants=dict(before=before,after=after),resource=guard.check()))


if __name__=='__main__':main()
