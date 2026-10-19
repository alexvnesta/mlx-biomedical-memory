# Replaying the comparison after readout repair

Use a fresh clone for new output directories. Existing runs are deliberately
preserved. Published adapters and raw outputs are in `artifacts/`; the native
base weights and large derived hidden-state caches are not redistributed.

## Portable grading replay

The CPU CI job replays the recorded outputs, paths, source/checkpoint bindings,
qualification decisions, data generation and report payload. It does not rerun
neural generation or training. Its dependencies and exact commands are in
[the workflow](../.github/workflows/protocol-tests.yml). On the prepared Mac:

```bash
.venv/bin/python scripts/verify_artifacts.py
```

## Neural replay from the published checkpoint

On Apple silicon, install the pinned environment with
`uv sync --python 3.13 --extra dev`. Cache the native base at the pinned revision once:

```python
from huggingface_hub import snapshot_download
from engram_poc.longcat import MODEL_ID, MODEL_REVISION
snapshot_download(MODEL_ID, revision=MODEL_REVISION,
                  allow_patterns=['*.json', '*.jinja', '*.safetensors'])
```

The learned reader uses the native base and published reader weights; it does
not require downloading the separate EngramEdit MQuAKE edited-state release.
Replay the qualified memory reader without retraining:

```bash
.venv/bin/python scripts/evaluate_decisive_reader.py \
  --training artifacts/composition-memory-2401 \
  --features artifacts/composition-features-v1 \
  --checkpoint-name selected.safetensors --include-support-readouts \
  --output runs/composition-memory-replay
```

Substitute `artifacts/composition-text-2401` or `composition-text-2402` and a new
output directory for the text controls. The evaluator needs the published corpus
and checkpoint, not the large training hidden-state files. Other fitted readers
failed the declared development gate; they are not eligible reasoning tests.

For the supplied-path sign test, regenerate input-only prefixes once, then replay
an eligible published reader:

```bash
.venv/bin/python scripts/cache_oracle_composition.py \
  --output runs/composition-oracle-replay-features
.venv/bin/python scripts/evaluate_oracle_composition.py \
  --training artifacts/composition-memory-2401 \
  --features runs/composition-oracle-replay-features \
  --output runs/composition-memory-replay-oracle
```

## Training and qualification

The exact executed training sources, configuration, native-weight hashes,
per-update losses, generated development outputs, BF16 candidates and selections
are archived in each `artifacts/composition-*` directory. The
[readout repair](READOUT-REPAIR-RESULTS.md) supplies the prerequisite checkpoint.

In a fresh clone, initialize the local prerequisite and training corpus:

```bash
mkdir -p runs
cp -R artifacts/readout-curriculum-v5 runs/readout-curriculum-v5
.venv/bin/python scripts/prepare_composition_experiment.py \
  --readout artifacts/readout-repair-test-v5
.venv/bin/python scripts/cache_composition_features.py
```

`prepare_composition_experiment.py` builds the fixed corpus from a completed
readout prerequisite; `cache_composition_features.py` computes only causal
training/development states. `run_composition_experiment.py --development-only`
fits the frozen encoder and fresh text controls. A failed declared gate produces
a nonzero exit and a preserved gate receipt; it is not permission to lower the
threshold. Inspect that receipt separately from runtime/resource failures.
`run_joint_composition_experiment.py --development-only` adds the registered
joint arm and records the eligibility of each reader/seed.

Publish selected checkpoint and corpus/source hashes before any final outputs.
Only then run the joint controller without `--development-only`, followed by
`report_composition.py`, `publish_composition_artifacts.py`, and
`verify_artifacts.py`. The controller reads the existing closed development gates
and tests eligible readers only. It never selects from final results.

The scripts serialize model processes with a kernel lease, limit Metal memory,
and stop at warning pressure or excess swap growth. Let an exited process finish
and wait for normal pressure before resuming. The controllers resume incomplete
evaluations with unchanged source/checkpoint hashes and retain saved predictions.
They do not stop other sessions. Hardware and numerical differences can change a
new fit's selection; the published receipts describe the measured Mac run, not a
guarantee of identical future training. New scientific comparisons require fresh
reserved cases rather than tuning on these now-public test outcomes.
