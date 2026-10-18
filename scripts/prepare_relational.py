"""Build the custom richer benchmark or verify its exact public replay bytes."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil

from engram_poc.relational_data import build


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replay',action='store_true')
    parser.add_argument('--cache-encoder',action='store_true')
    args=parser.parse_args();out=Path('data/processed/relational-v1')
    if args.replay:
        source=Path('artifacts/relational-fit-v1');out.mkdir(parents=True,exist_ok=True)
        raw=(source/'dataset.json').read_bytes();config=json.loads((source/'config.json').read_text())
        assert hashlib.sha256(raw).hexdigest()==config['dataset_sha256']
        (out/'dataset.json').write_bytes(raw);print('Verified exact public dataset.')
    else:build(out)
    if args.cache_encoder:
        from transformers import AutoModel,AutoTokenizer
        model='BAAI/bge-small-en-v1.5';revision='5c38ec7c405ec4b44b94cc5a9bb96e735b38267a'
        AutoTokenizer.from_pretrained(model,revision=revision)
        AutoModel.from_pretrained(model,revision=revision)
        print('Cached the pinned CPU encoder.')


if __name__=='__main__':main()
