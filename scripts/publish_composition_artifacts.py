"""Publish composition receipts, closed fits and exact evaluated checkpoints."""
import hashlib
import json
from pathlib import Path
import shutil


def main():
    public=Path('artifacts')
    for run in sorted(Path('runs').glob('composition-*')):
        if not run.is_dir():continue
        is_root=run.name=='composition-v1';is_features=run.name=='composition-features-v1'
        if not is_root and not (run/'summary.json').exists():continue
        target=public/run.name;target.mkdir(exist_ok=True)
        for source in run.rglob('*'):
            if not source.is_file() or 'features' in source.relative_to(run).parts:continue
            if source.name in ['last.safetensors','optimizer.safetensors','state.json']:continue
            if source.suffix not in ['.json','.jsonl','.py','.md','.safetensors','.log']:continue
            dest=target/source.relative_to(run);dest.parent.mkdir(exist_ok=True,parents=True);shutil.copyfile(source,dest)
        inventory={p.name:dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size,
            published=(target/p.name).exists()) for p in run.glob('*.safetensors')}
        if inventory:(target/'checkpoint-inventory.json').write_text(json.dumps(inventory,indent=2)+'\n')
        log=Path(str(run)+'.log')
        if log.exists() and not is_root:shutil.copyfile(log,target/'execution.log')
        if is_root:
            shutil.copyfile('scripts/run_composition_experiment.py',target/'controller.py')
            shutil.copyfile('docs/COMPOSITION-PROTOCOL.md',target/'protocol.md')
            if (run/'development-gate.json').exists():shutil.copyfile('runs/composition-development.log',target/'development.log')
    for name in ['composition.html','composition-summary.json','composition-comparison.png']:
        path=Path('runs')/name
        if path.exists():shutil.copyfile(path,public/name)
    inventory={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(public.rglob('*')) if p.is_file() and p.name!='SHA256SUMS.json'}
    (public/'SHA256SUMS.json').write_text(json.dumps(inventory,indent=2)+'\n')
    print(json.dumps(dict(checksummed=len(inventory))))


if __name__=='__main__':main()
