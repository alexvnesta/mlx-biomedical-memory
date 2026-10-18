"""Verify MLX token IDs against the official LongCat tokenizer on MQuAKE inputs."""

import argparse
import hashlib
import json
from pathlib import Path

import truststore
truststore.inject_into_ssl()

from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

from engram_poc.longcat import MODEL_ID, MODEL_REVISION


OFFICIAL_REVISION = "b62b68827ead0b7fef3ba98b57f18484acaaec06"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, default=Path("data/processed/longcat"))
    parser.add_argument("--output", type=Path, default=Path("runs/longcat-tokenizer-parity.json"))
    args = parser.parse_args()
    official_path = hf_hub_download("meituan-longcat/LongCat-Flash-Lite", "tokenizer.json", revision=OFFICIAL_REVISION)
    mlx_path = hf_hub_download(MODEL_ID, "tokenizer.json", revision=MODEL_REVISION)
    official, converted = Tokenizer.from_file(official_path), Tokenizer.from_file(mlx_path)
    records = json.loads((args.state / "MQuAKE-CF-3k-v2.json").read_text())
    texts = []
    for record in records:
        texts.extend(record["questions"])
        for edit in record["requested_rewrite"]:
            texts.extend([edit["prompt"].format(edit["subject"]), edit["subject"], edit["target_new"]["str"]])
    mismatches = sum(official.encode(text, add_special_tokens=False).ids != converted.encode(text, add_special_tokens=False).ids for text in texts)
    result = dict(official_revision=OFFICIAL_REVISION,
        official_tokenizer_sha256=hashlib.sha256(Path(official_path).read_bytes()).hexdigest(),
        mlx_revision=MODEL_REVISION, mlx_tokenizer_sha256=hashlib.sha256(Path(mlx_path).read_bytes()).hexdigest(),
        vocab_exact=official.get_vocab() == converted.get_vocab(),
        texts_compared=len(texts), tokenization_mismatches=mismatches,
        official_postprocessor=json.loads(Path(official_path).read_text()).get("post_processor"),
        mlx_postprocessor=json.loads(Path(mlx_path).read_text()).get("post_processor"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))
    if not result["vocab_exact"] or mismatches:
        raise RuntimeError("Tokenizer parity failed")


if __name__ == "__main__":
    main()
