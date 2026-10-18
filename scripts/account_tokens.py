"""Count exact prompt tokens for released benchmarks; not a latency benchmark."""

import json
from pathlib import Path
from statistics import mean

from transformers import AutoTokenizer
from engram_poc.benchmark import load_manifest
from engram_poc.experiment import MODEL_ID, MODEL_REVISION

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
out = []
for path in sorted(Path("artifacts").glob("*/dataset.json")):
    config = json.loads((path.parent / "config.json").read_text())
    if not config.get("source_sha256"):
        # Older diagnostic pilots used different prompt/projection formats.
        continue
    data = load_manifest(path)
    counts = {}
    for mode, oracle in [("memory_or_base", False), ("text_facts", True)]:
        lengths = []
        for ex in data["test"]:
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": ex.prompt(oracle)}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            lengths.append(len(tokenizer.encode(text, add_special_tokens=False)))
        counts[mode] = {"mean": mean(lengths), "min": min(lengths), "max": max(lengths)}
    out.append({"run": path.parent.name, "test_prompt_tokens": counts})
Path("artifacts/token-accounting.json").write_text(
    json.dumps(
        {
            "model": MODEL_ID,
            "revision": MODEL_REVISION,
            "scope": "deterministic prompt lengths only; not throughput, correctness, or amortized efficiency",
            "runs": out,
        },
        indent=2,
    )
    + "\n"
)
print(json.dumps(out, indent=2))
