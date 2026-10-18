"""Export all completed runs, including negative pilots, without large weights or raw source data."""

import hashlib
import json
from pathlib import Path
import shutil
from statistics import mean

destination = Path("artifacts")
destination.mkdir(exist_ok=True)
rows = []
for path in sorted(Path("runs").glob("*/results.json")):
    run = path.parent
    target = destination / run.name
    target.mkdir(exist_ok=True)
    for name in [
        "results.json",
        "config.json",
        "selection.json",
        "dataset.json",
        "training.jsonl",
        "addresses.json",
    ]:
        if (run / name).exists():
            shutil.copy2(run / name, target / name)
    r = json.loads(path.read_text())
    config = json.loads((run / "config.json").read_text())
    conditions = r["conditions"]
    rows.append(
        {
            "run": run.name,
            "seed": config["seed"],
            "dimension": config["dimension"],
            "steps": config["steps"],
            "arax": bool(config.get("arax_snapshot")),
            "initialization": config.get("memory_initialization", "random"),
            "composition": {k: v["accuracy"] for k, v in conditions.items()},
            "memory_atomic_recall": r["atomic_recall"]["memory"]["accuracy"],
            "paraphrase": r["paraphrase_memory"]["accuracy"],
            "test_n": conditions["memory"]["n"],
            "test_components": r.get("test_components"),
            "unique_questions": r.get("test_unique_questions"),
            "backbone_unchanged": r["backbone"]["unchanged"],
            "runtime": r["runtime"],
        }
    )
(destination / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
lines = [
    "# Measured results",
    "",
    "All completed local runs are retained below. These are small exploratory experiments,",
    "not evidence of a rare-disease discovery or general superiority over LLM+KG.",
    "",
    "| Run | Base | Supplied text facts | Latent memory | Shuffled | Atomic recall with memory | Paraphrase |",
    "|---|---:|---:|---:|---:|---:|---:|",
]
for r in rows:
    c = r["composition"]

    def pct(value):
        return f"{100 * value:.1f}%"

    lines.append(
        f"| [{r['run']}](../artifacts/{r['run']}/results.json) | "
        + " | ".join(
            pct(v)
            for v in [
                c["base"],
                c["text_facts"],
                c["memory"],
                c["shuffled_memory"],
                r["memory_atomic_recall"],
                r["paraphrase"],
            ]
        )
        + " |"
    )
lines += [
    "",
    "## Interpretation",
    "",
    "The initial pilots used randomly initialized memory rows. Low held-out atomic recall",
    "limits conclusions about reasoning: a model cannot reliably compose premises that its",
    "memory has not learned. A small composition score above the disabled control is not",
    "sufficient evidence of systematic generalization. The supplied-text comparator is",
    "stronger than end-to-end retrieval because it receives the exact required premises.",
    "",
    "The first smoke run used a plain completion probe and four optimizer steps. It is an",
    "execution check, not an architectural comparison. Later runs use chat-formatted A/B",
    "probes. Early pilots projected only the final hidden state; the corrected wrapper",
    "preserves the stock projection shape to make memory-disabled computation identical.",
    "",
    "The synthetic tests have 16 held-out components unless the run is explicitly a smoke",
    "test. ARAX rows include repeated positives for class weighting: count unique questions",
    "and components, not rows, when assessing uncertainty. Run configurations and raw",
    "per-example predictions are linked above. The code and data hashes are recorded in",
    "later run configurations. Early pilots without a source hash remain exploratory.",
    "",
    "See [protocol](PROTOCOL.md) for what the experiment does and does not establish.",
]
fresh = [r for r in rows if r["run"].startswith("zero-init-seed")]
if fresh:
    lines += [
        "",
        "## Fresh-seed summary",
        "",
        f"The three fixed-recipe zero-initialization runs have mean held-out composition "
        f"accuracy {100 * mean(r['composition']['memory'] for r in fresh):.1f}% with latent memory, "
        f"versus {100 * mean(r['composition']['text_facts'] for r in fresh):.1f}% with supplied text facts. "
        "This does not demonstrate a reasoning advantage. Each synthetic run has only 16 held-out components.",
        "",
        "Prompt lengths are recorded separately in [token accounting](../artifacts/token-accounting.json). "
        "Fewer prompt tokens alone are not an efficiency benefit when answer accuracy is lower. "
        "No latency or amortized-training-cost advantage has been established.",
    ]
for r in [row for row in rows if row["arax"]]:
    lines += [
        "",
        "## ARAX-derived snapshot task",
        "",
        f"The `{r['run']}` run has {r['test_n']} weighted test rows, "
        f"{r['unique_questions']} unique supplied edge-pair questions, and "
        f"{r['test_components']} held-out Rett-syndrome path components. "
        f"Latent-memory membership composition is {100 * r['composition']['memory']:.1f}%, "
        f"versus {100 * r['composition']['text_facts']:.1f}% with exact text facts. "
        f"Atomic recall with memory is {100 * r['memory_atomic_recall']:.1f}%.",
        "",
        "This is a conjunction of recorded/unobserved predicate-level edge memberships, "
        "not sign inference, hidden-path discovery, a treatment prediction, or validation of a new biomedical relationship. "
        "Weak atomic recall limits conclusions about a better memory interface.",
    ]
Path("docs/RESULTS.md").write_text("\n".join(lines) + "\n")
hashes = {
    str(p): hashlib.sha256(p.read_bytes()).hexdigest()
    for p in destination.rglob("*")
    if p.is_file() and p.name != "SHA256SUMS.json"
}
(destination / "SHA256SUMS.json").write_text(json.dumps(hashes, indent=2) + "\n")
print(json.dumps(rows, indent=2))
