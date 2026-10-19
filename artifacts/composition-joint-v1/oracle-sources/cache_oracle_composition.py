"""Cache oracle-path input prefixes; no output sign or answer is included."""
import argparse
import hashlib
import json
from pathlib import Path

import mlx.core as mx

from engram_poc.decisive_model import load_reader
from engram_poc.decisive_prompts import encode_prompt
from engram_poc.decisive_data import fields
from engram_poc.longcat_runtime import save,append,snapshot_source
from engram_poc.resource_guard import ResourceGuard
from cache_decisive_features import digest_frozen
from oracle_composition_data import build


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('runs/composition-oracle-features-v1'))
    p.add_argument('--resume',action='store_true');args=p.parse_args();out=args.output
    if out.exists() and not args.resume:raise RuntimeError('Preserve oracle prefixes')
    out.mkdir(exist_ok=True);(out/'features').mkdir(exist_ok=True)
    cases=build()
    if not args.resume:
        save(out/'cases.json',cases)
        save(out/'config.json',dict(source_sha256=snapshot_source(__file__,out/'source'),
            cases_sha256=hashlib.sha256((out/'cases.json').read_bytes()).hexdigest(),
            protocol_sha256=hashlib.sha256(Path('docs/COMPOSITION-ORACLE-PATH-PROTOCOL.md').read_bytes()).hexdigest(),
            producer_sha256=hashlib.sha256(Path('scripts/oracle_composition_data.py').read_bytes()).hexdigest(),
            dependency_sha256=hashlib.sha256(Path('scripts/cache_decisive_features.py').read_bytes()).hexdigest(),
            semantics='Input-only prefix with supplied path IDs; no output sign or answer token'))
        (out/'oracle_composition_data.py').write_bytes(Path('scripts/oracle_composition_data.py').read_bytes())
        (out/'protocol.md').write_bytes(Path('docs/COMPOSITION-ORACLE-PATH-PROTOCOL.md').read_bytes())
        (out/'cache_decisive_features.py').write_bytes(Path('scripts/cache_decisive_features.py').read_bytes())
    else:
        cfg=json.loads((out/'config.json').read_text());assert json.loads((out/'cases.json').read_text())==cases
        assert hashlib.sha256((out/'cases.json').read_bytes()).hexdigest()==cfg['cases_sha256']
        source=hashlib.sha256()
        for path in [Path(__file__),*sorted(Path('engram_poc').glob('*.py'))]:source.update(path.name.encode());source.update(path.read_bytes())
        assert source.hexdigest()==cfg['source_sha256']
        assert hashlib.sha256(Path('scripts/oracle_composition_data.py').read_bytes()).hexdigest()==cfg['producer_sha256']
        assert hashlib.sha256(Path('scripts/cache_decisive_features.py').read_bytes()).hexdigest()==cfg['dependency_sha256']
        assert hashlib.sha256(Path('docs/COMPOSITION-ORACLE-PATH-PROTOCOL.md').read_bytes()).hexdigest()==cfg['protocol_sha256']
    guard=ResourceGuard();model,tok=load_reader('memory');before=digest_frozen(model)
    ids={label:tok.encode(label,add_special_tokens=False) for label in ['UP','DOWN']}
    assert all(len(v)==1 for v in ids.values());save(out/'sign-token-ids.json',ids)
    rows=[json.loads(x) for x in (out/'manifest.jsonl').read_text().splitlines()] if (out/'manifest.jsonl').exists() else []
    done={(r['case_id'],r['mode']) for r in rows};shared={}
    for r in rows:
        if r['mode']=='memory':shared[r['world_id']]=dict(path=r['path'],feature_sha256=r['feature_sha256'],token_sha256=r['token_sha256'],shape=r['shape'])
    for original in cases:
        case=original['case']
        scaffold='{"paths":[{"edges":'+json.dumps(original['path_ids'],separators=(',',':'))+',"sign":"'
        partial=tok.encode(scaffold,add_special_tokens=False)
        for label,token in ids.items():assert tok.encode(scaffold+label,add_special_tokens=False)==partial+token
        for mode in ['text','memory']:
            if (case['id'],mode) in done:continue
            e=encode_prompt(tok,case,mode);tokens=e['ids']+partial
            token_sha=hashlib.sha256(json.dumps(tokens).encode()).hexdigest()
            common=shared.get(original['world_id']) if mode=='memory' else None
            if common:
                assert common['token_sha256']==token_sha
            else:
                h=model.prefix(mx.array([tokens]));mx.eval(h)
                path=out/'features'/((original['world_id'] if mode=='memory' else case['id'])+'-'+mode+'.safetensors')
                mx.save_safetensors(str(path),dict(hidden=h))
                common=dict(path=str(path),feature_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),token_sha256=token_sha,shape=list(h.shape))
                if mode=='memory':shared[original['world_id']]=common
                del h;mx.clear_cache()
            row=dict(case_id=case['id'],world_id=original['world_id'],mode=mode,depth=case['depth'],
                nuisance=original['nuisance'],expected_sign=original['expected_sign'],fields=fields(case['records']),
                positions=e['placements'],path_ids=original['path_ids'],input_tokens=len(tokens),**common)
            append(out/'manifest.jsonl',row);rows.append(row)
            if len(rows)%64==0:print(json.dumps(dict(prefixes=len(rows),resource=guard.check())),flush=True)
    after=digest_frozen(model);assert before==after
    save(out/'summary.json',dict(completed=True,rows=len(rows),unique_memory_prefixes=len(shared),
        frozen_constants=dict(before=before,after=after),resource=guard.check()))


if __name__=='__main__':main()
