"""Bounded LongCat native-row offload qualification before reader training."""

import json
from pathlib import Path
import subprocess
import time

import mlx.core as mx
from huggingface_hub import snapshot_download
from mlx_lm.utils import load_model

from engram_poc.disk_embeddings import offload_ngram_tables
from engram_poc.longcat import MODEL_ID,MODEL_REVISION


def main():
    mx.set_memory_limit(29_000_000_000);mx.set_cache_limit(64*1024**2)
    path=Path(snapshot_download(MODEL_ID,revision=MODEL_REVISION,local_files_only=True,
        allow_patterns=['*.json','*.jinja','*.safetensors']))
    print('Lazy native model load',flush=True);model,_=load_model(path,lazy=True)
    print('Qualifying disk-backed native rows',flush=True)
    receipt=offload_ngram_tables(model,path)
    print('Loading remaining parameters',flush=True);mx.eval(model.parameters())
    start=time.perf_counter();logits=model(mx.array([[1,121,414,390]]));mx.eval(logits)
    result=dict(model=MODEL_ID,revision=MODEL_REVISION,tables=receipt,
        resident_metal_gb=mx.get_active_memory()/1e9,peak_metal_gb=mx.get_peak_memory()/1e9,
        forward_seconds=time.perf_counter()-start,finite=mx.all(mx.isfinite(logits)).item(),
        swap=subprocess.check_output(['sysctl','vm.swapusage'],text=True).strip())
    Path('runs/decisive-runtime-probe.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
