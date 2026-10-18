"""Download pinned references and convert the authors' safe-loaded edit state.

PyTorch is used only in this preparation process, not for MLX inference.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import urllib.request

import truststore

truststore.inject_into_ssl()

import mlx.core as mx
import numpy as np
import torch
from huggingface_hub import hf_hub_download, snapshot_download

from engram_poc.longcat import (
    MODEL_ID, MODEL_REVISION, STATE_ID, STATE_REVISION, UPSTREAM_COMMIT, MQUAKE_COMMIT,
)
from engram_poc.longcat_keys import LENGTHS


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/processed/longcat"))
    parser.add_argument("--download-model", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Choose an empty preparation directory")
    args.output.mkdir(parents=True, exist_ok=True)
    state_path = hf_hub_download(STATE_ID, "engramedit_state.pt", revision=STATE_REVISION)
    payload = torch.load(state_path, map_location="cpu", weights_only=True)
    if payload.get("ngram_key_lengths") != list(LENGTHS) or set(payload["memory"]) != set(LENGTHS):
        raise ValueError("Unexpected released state schema")
    manifest = dict(
        format="engramedit-mlx-v1", model_id=MODEL_ID, model_revision=MODEL_REVISION,
        state_id=STATE_ID, state_revision=STATE_REVISION, source_state_sha256=sha(state_path),
        upstream_commit=UPSTREAM_COMMIT, mquake_commit=MQUAKE_COMMIT,
        source_model_name=payload.get("model_name"),
        processed_request_ids=payload.get("processed_request_ids", []),
        skipped_request_ids=payload.get("skipped_request_ids", []), keys={},
        prepared_utc=datetime.now(timezone.utc).isoformat(), downloads=[],
    )
    arrays = {}
    for length in LENGTHS:
        table = payload["memory"][length]
        keys = sorted(table)
        values = [np.zeros(3072, dtype=np.float32)]
        for key in keys:
            if len(key) != length or any(type(x) is not int or not 0 <= x < 131072 for x in key):
                raise ValueError("Invalid released key")
            tensor = table[key]
            if tensor.shape != (3072,) or not tensor.is_floating_point() or not torch.isfinite(tensor).all():
                raise ValueError("Invalid released tensor")
            values.append(tensor.float().numpy())
        manifest["keys"][str(length)] = keys
        arrays[str(length)] = mx.array(np.stack(values))
    updates_path = args.output / "updates.safetensors"
    mx.save_safetensors(str(updates_path), arrays)
    manifest["updates_sha256"] = sha(updates_path)
    downloads = {
        "MQuAKE-CF-3k-v2.json": f"https://raw.githubusercontent.com/princeton-nlp/MQuAKE/{MQUAKE_COMMIT}/datasets/MQuAKE-CF-3k-v2.json",
        "multihop-prompts.txt": f"https://raw.githubusercontent.com/ModalityDance/EngramEdit/{UPSTREAM_COMMIT}/dsets/mquake_prompts/multihop-prompts.txt",
        "multihop-cot-prompts.txt": f"https://raw.githubusercontent.com/ModalityDance/EngramEdit/{UPSTREAM_COMMIT}/dsets/mquake_prompts/multihop-cot-prompts.txt",
    }
    for filename, url in downloads.items():
        with urllib.request.urlopen(url, timeout=45) as response:
            content = response.read()
        path = args.output / filename
        path.write_bytes(content)
        manifest["downloads"].append(dict(filename=filename, url=url, sha256=sha(path)))
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if args.download_model:
        snapshot_download(MODEL_ID, revision=MODEL_REVISION, allow_patterns=["*.json", "*.jinja", "*.safetensors"])
    print(json.dumps(dict(output=str(args.output), keys={k: len(v) for k, v in manifest["keys"].items()}, processed_requests=len(manifest["processed_request_ids"])), indent=2))


if __name__ == "__main__":
    main()
