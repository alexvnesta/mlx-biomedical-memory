"""Bounded real-model feasibility probe before a quantized editing experiment."""
import json
from pathlib import Path
import time

import mlx.core as mx
from huggingface_hub import snapshot_download
from mlx_lm.utils import load_model
from mlx_lm.tokenizer_utils import TokenizerWrapper
from transformers import PreTrainedTokenizerFast

from engram_poc.longcat import MODEL_ID, MODEL_REVISION
from engram_poc.longcat_editor import install_editable, fit_target, allocate_updates

mx.set_memory_limit(48_000_000_000)
mx.set_cache_limit(128*1024**2)
path=snapshot_download(MODEL_ID,revision=MODEL_REVISION,local_files_only=True,
    allow_patterns=["*.json","*.jinja","*.safetensors"])
print("Loading model",flush=True)
model,config=load_model(Path(path))
tokenizer=TokenizerWrapper(PreTrainedTokenizerFast.from_pretrained(path,local_files_only=True),eos_token_ids=[2])
memory=install_editable(model)
request=dict(subject="Paris",prompt="The country containing {} is",target="Japan")
def record(row):print(json.dumps(row),flush=True)
start=time.perf_counter()
result=fit_target(model,memory,tokenizer,request,steps=3,log=record)
updates,allocation=allocate_updates([result])
memory.add_updates(updates)
out=Path("runs/longcat-gradient-probe");out.mkdir(exist_ok=True)
summary=dict(history=result["history"],allocation=allocation,wall_seconds=time.perf_counter()-start,
    peak_metal_gb=mx.get_peak_memory()/1e9)
(out/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
print(json.dumps(summary),flush=True)
