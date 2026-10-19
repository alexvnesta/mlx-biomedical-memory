"""Publish closed readout runs, including failures, without copying native weights."""
import hashlib
import json
from pathlib import Path
import shutil


def main():
    public=Path('artifacts');copied=[]
    for run in sorted(Path('runs').glob('readout-*')):
        if not run.is_dir() or not any((run/name).exists() for name in ['summary.json','interruption.json']):continue
        target=public/run.name;target.mkdir(exist_ok=True)
        for source in run.rglob('*'):
            if not source.is_file() or source.name in ['last.safetensors','optimizer.safetensors']:continue
            if source.suffix not in ['.json','.jsonl','.py','.md','.safetensors','.log']:continue
            destination=target/source.relative_to(run);destination.parent.mkdir(exist_ok=True,parents=True)
            shutil.copyfile(source,destination);copied.append(str(destination))
        inventory={source.name:dict(sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            published=(target/source.name).exists(),bytes=source.stat().st_size)
            for source in run.glob('*.safetensors')}
        if inventory:(target/'checkpoint-inventory.json').write_text(json.dumps(inventory,indent=2)+'\n')
        log=Path(str(run)+'.log')
        if log.exists():shutil.copyfile(log,target/'execution.log')
    for name in ['readout.html','readout-summary.json','readout-resource-incident.json']:
        source=Path('runs')/name
        if source.exists():shutil.copyfile(source,public/name)
    inventory={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(public.rglob('*')) if p.is_file() and p.name!='SHA256SUMS.json'}
    (public/'SHA256SUMS.json').write_text(json.dumps(inventory,indent=2)+'\n')
    print(json.dumps(dict(copied=len(copied),checksummed=len(inventory))))


if __name__=='__main__':main()
