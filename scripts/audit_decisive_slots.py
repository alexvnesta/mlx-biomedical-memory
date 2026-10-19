"""Independently align injected memory slots against native tokenizer offsets."""
import hashlib
import json
from pathlib import Path
from huggingface_hub import snapshot_download
from transformers import PreTrainedTokenizerFast
from mlx_lm.tokenizer_utils import TokenizerWrapper
from engram_poc.longcat import MODEL_ID,MODEL_REVISION
from engram_poc.decisive_prompts import encode_prompt


def main():
    model=snapshot_download(MODEL_ID,revision=MODEL_REVISION,local_files_only=True,allow_patterns=['*.json','*.jinja','*.safetensors'])
    tokenizer=TokenizerWrapper(PreTrainedTokenizerFast.from_pretrained(model,local_files_only=True),eos_token_ids=[2])
    dataset=Path('data/processed/decisive-v1/dataset.json');depth=Path('data/processed/decisive-v1/depth-only.json')
    corpus=json.loads(dataset.read_text());cases=[c for group in corpus['splits'].values() for c in group]+corpus['interventions']+json.loads(depth.read_text())['cases'];slots=0
    for case in cases:
        encoded=encode_prompt(tokenizer,case,'memory');offsets=tokenizer._tokenizer(encoded['prompt'],add_special_tokens=False,return_offsets_mapping=True)['offset_mapping'];cursor=0
        for record,position in zip(case['records'],encoded['placements']):
            mark=encoded['prompt'].index(record['id']+': @',cursor)+len(record['id']+': ');cursor=mark+1
            assert offsets[position][0]<=mark<offsets[position][1],(case['id'],record['id'],position)
            slots+=1
    output=Path('runs/decisive-slot-audit.json');output.write_text(json.dumps(dict(completed=True,slots=slots,cases=len(cases),all_exact=True,
        dataset_sha256=hashlib.sha256(dataset.read_bytes()).hexdigest(),depth_sha256=hashlib.sha256(depth.read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2)+'\n')
    print(output.read_text())


if __name__=='__main__':main()
