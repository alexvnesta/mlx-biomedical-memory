"""Publish reproducible receipts, generated outcomes and selected trained readers."""
import hashlib
import json
from pathlib import Path
import shutil


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    public=Path('artifacts');copied=[]
    for run in sorted(Path('runs').glob('decisive-*')):
        if not run.is_dir() or not (run/'config.json').exists() and not (run/'summary.json').exists():continue
        if run.name=='decisive-features-v1':continue
        target=public/run.name;target.mkdir(exist_ok=True)
        for source in run.rglob('*'):
            if not source.is_file() or 'features' in source.parts:continue
            if source.suffix=='.safetensors' and not (source.name in {'selected.safetensors','trained-selected.safetensors','initial.safetensors'} or source.name.startswith(('best-','final-'))):continue
            if source.name=='state.json':continue # Optimizer-resume bookkeeping is not a published checkpoint.
            if source.suffix not in {'.json','.jsonl','.py','.md','.safetensors'}:continue
            destination=target/source.relative_to(run);destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(source,destination);copied.append(str(destination))
        checkpoints={p.name:dict(sha256=digest(p),bytes=p.stat().st_size,
            published=(target/p.name).exists(),
            role='Exact evaluated BF16 reader' if p.name in {'selected.safetensors','trained-selected.safetensors'} else 'BF16 training candidate' if p.name.startswith(('initial','best','final')) else 'Local FP32 optimizer/resume state; regenerate with archived training source')
            for p in run.glob('*.safetensors')}
        if checkpoints:(target/'checkpoint-inventory.json').write_text(json.dumps(checkpoints,indent=2)+'\n')
        log=Path(str(run)+'.log')
        if log.exists():shutil.copyfile(log,target/'execution.log')
    for name in ['decisive-development-gate.json','decisive-development-gate-before-retention.json','decisive-slot-audit.json','decisive-slot-audit-source.py','decisive-parameter-audit.json','decisive-parameter-audit-source.py','decisive-summary.json']:
        source=Path('runs')/name
        if source.exists():shutil.copyfile(source,public/name)
    for name in ['decisive.html','decisive-study.log','decisive-candidate-retention.log','decisive-finish.log']:
        source=Path('runs')/name
        if source.exists():shutil.copyfile(source,public/name)
    inventory={str(p):digest(p) for p in sorted(public.rglob('*')) if p.is_file() and p.name!='SHA256SUMS.json'}
    (public/'SHA256SUMS.json').write_text(json.dumps(inventory,indent=2)+'\n')
    print(json.dumps(dict(copied=len(copied),checksummed=len(inventory))))


if __name__=='__main__':main()
