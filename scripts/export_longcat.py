"""Publish every completed LongCat replay and check receipt, without weights."""

import hashlib
import json
from pathlib import Path
import shutil


root = Path("artifacts")
completed = []
lines = [
    "# LongCat MLX replay: measured results", "",
    "Exploratory replay of the authors' MQuAKE editing state on 4-bit MLX",
    "LongCat-Flash-Lite. This is not new biomedical training or an exact BF16",
    "paper reproduction. See [protocol and limitations](LONGCAT.md).", "",
    "The six-case smoke test used one question per case. The later 30-case replay",
    "uses all three question variants and includes those initial six cases. It was",
    "expanded after seeing the smoke result, without changing the checkpoint or",
    "generation recipe. The overlapping runs are not independent replications.", "",
    "| Run | Style | Condition | Updated-answer question accuracy | Cases with any updated answer |",
    "|---|---|---|---:|---:|",
]
for run in sorted(Path("runs").glob("longcat-mquake-*")):
    if not run.is_dir() or not (run / "summary.json").exists():
        continue
    destination = root / run.name
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("config.json", "verification.json", "summary.json", "predictions.jsonl"):
        shutil.copy2(run / name, destination / name)
    if (run / "source").exists():
        shutil.copytree(run / "source", destination / "source", dirs_exist_ok=True)
    result = json.loads((run / "summary.json").read_text())
    completed.append((run.name, result))
    for style, conditions in result["conditions"].items():
        for condition, values in conditions.items():
            accuracy = values["updated_answer_accuracy"]
            case_accuracy = values.get("case_any_updated_accuracy", accuracy)
            lines.append(f"| [{run.name}](../artifacts/{run.name}/summary.json) | {style} | {condition} | "
                f"{100*accuracy:.1f}% ({values['n']} questions) | {100*case_accuracy:.1f}% |")
for name in ("longcat-reference-parity.json", "longcat-tokenizer-parity.json"):
    if (Path("runs") / name).exists():
        receipt = json.loads((Path("runs") / name).read_text())
        receipt.pop("official_tokenizer_config", None)
        (root / name).write_text(json.dumps(receipt, indent=2)+"\n")
lines += ["", "## Validation and interpretation", "",
    "The disabled wrapper matches tested native-model logits exactly. Native and",
    "edited embeddings match exactly between full and chunked inputs. The actual",
    "published residuals also match the pinned upstream PyTorch wrapper exactly",
    "in BF16 on a 49-token test with identical synthetic base outputs, EOS boundaries",
    "and two-token chunks. This isolates residual replay, not full-backbone equivalence.", "",
    "MLX and official tokenizers have identical vocabularies and matching token IDs",
    "on all 27,363 tested questions, atomic prompts, subjects and new targets.", "",
    "Full-decoder logits differ between full and cached execution on the tested key:",
    "maximum differences are 0.25 without edits and 0.875 with edits, although the",
    "edited key's next-token argmax matched. The embedding cache itself was exact.",
    "Framework arithmetic, quantization and transfer of BF16-trained residuals to a",
    "4-bit backbone remain relevant uncertainties; weak transfer is not a general",
    "refutation of EngramEdit. This replay does not isolate their effects.", "",
    "Text edits supply revised facts, not every reasoning premise or actual retrieval.",
    "Base LongCat retains its native pretrained n-gram memory. Original-answer scores",
    "are retained too: the unedited model should not know fictional updates.", "",
    "Timing uses fixed condition order and different completion lengths. It does not",
    "establish a speed advantage. Peak Metal allocation is not total system memory.",
    "System swap grew during model loading on this busy 64 GB Mac; model fit alone",
    "does not establish comfortable headroom for longer contexts or edit optimization.", "",
    "Target optimization, frequency regularization and fitting new ARAX edits are",
    "not implemented in the Mac replay. All completed outcomes are retained.", "",
    "Receipts: [residual parity](../artifacts/longcat-reference-parity.json),",
    "[tokenizer parity](../artifacts/longcat-tokenizer-parity.json).", "",
]
for name, result in completed:
    lines += [f"`{name}`: **{result['peak_metal_gb']:.2f} GB peak Metal allocation**.", ""]
Path("docs/LONGCAT-RESULTS.md").write_text("\n".join(lines))
hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob("*")
          if p.is_file() and p.name != "SHA256SUMS.json"}
(root / "SHA256SUMS.json").write_text(json.dumps(hashes, indent=2)+"\n")
print(json.dumps(dict(exported=[name for name, _ in completed], artifacts=len(hashes)), indent=2))
