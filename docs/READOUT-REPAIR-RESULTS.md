# Memory readout repair

**Readout gate passed.** The original one-vector codec reconstructs 256/256 complete records (100.0%) from fresh ten-record banks, with subject–object pairs excluded from every training bank. No reasoning results are claimed in this repair phase.

Selected `readout-curriculum-v5`, step 2048, 10-record
development stage. Checkpoint SHA-256: `3d913d797cf7209590c88336a014ca44d6a6197fe422f2ecc92d01ef9f3bb3c1`.

| Condition | Complete-record exact | 95% Wilson interval |
|---|---:|---:|
| binding | 62/64 (96.9%) | 89.3–99.1% |
| change object | 63/64 (98.4%) | 91.7–99.7% |
| change sign | 64/64 (100.0%) | 94.3–100.0% |
| change source | 62/64 (96.9%) | 89.3–99.1% |
| change species | 62/64 (96.9%) | 89.3–99.1% |
| change subject | 62/64 (96.9%) | 89.3–99.1% |
| change variant | 63/64 (98.4%) | 91.7–99.7% |
| masked | 0/64 (0.0%) | 0.0–5.7% |
| normal | 256/256 (100.0%) | 98.5–100.0% |
| order | 63/64 (98.4%) | 91.7–99.7% |
| wrong bank | 0/64 (0.0%) | 0.0–5.7% |

Wrong-bank accuracy against the *actually injected* replacement values is
61/64. Original-target
accuracy for that ablation has a different meaning from normal readout accuracy.
The record ID is visible in the question; the other six fields are hidden.

## What failed, and what changed

Checkpoint loading, cached/direct logits, incremental-prefix logits, and FP32/BF16
generated answers agreed in the original failure audit (16 evaluations of eight
old banks). The earlier model produced valid JSON but incorrect hidden values.
It had seen only 128 targeted atomic records mixed with reasoning. Nonzero
gradients and low teacher-forced loss had not established usable memory access.

A fresh-bank full-record curriculum improved readout but plateaued at 28/32
development records. Object IDs remained wrong even with near-zero training
loss. Field-isolated questions removed preceding hidden values from each
field's teacher-forced target prefix. The corrected continuation used FP32
optimization, BF16 published evaluation, balanced field questions, full-record
questions, and a one/four/ten-record curriculum. Codec architecture and vector
count did not change. This is an engineering repair with multiple changes;
the experiment does not isolate one unique cause of the original failure.

The isolated-field audit measured 28/32 object values versus 19/32 within full
records for the same checkpoint. The next continuation added noisy past-value
teacher-forcing: corrupt preceding values with probability 0.5, preserve exact
token counts, and retain the true memory and correct next-token labels. All
training bank facts exclude the held-out pairs; noisy decoder contexts can
contain other alias combinations. Evaluation is normal free generation.

A development-only intervention kept memory fixed and varied the preceding subject over all 16 aliases. Teacher-forced object value tokens were exact in 497/512 contexts before repair and 512/512 after. All 15 earlier failures occurred when the preceding subject equaled the true object. These scores exclude the closing JSON delimiter and are diagnostic, not free-generation test accuracy.

The pair split puts all 16 self-relations in development and none in training or the normal test. Thus preceding subject = object was also a relation pattern absent from true training banks. With the correct preceding subject, self-relation object tokens improved from 12/19 to 19/19; non-self cases were already 13/13. This supports a decoder-history shortcut on a deliberately shifted pattern, rather than a single generic explanation for all original failures. These equality results are development diagnostics, not additional held-out accuracy.

The first scalar-loss implementation error and the short BF16 warm-start pilot
are retained as failed attempts. No fresh test outputs were used to select the
final checkpoint. All development outcomes, not just successful checkpoints,
are published.

## Limits

This establishes only the measured atomic readout behavior of this fitted
reader, on a finite vocabulary and banks of ten records. It is not a replication
of EngramEdit, a new biomedical finding, clinical validation, a scaling result,
or evidence of better reasoning than text, RAG, or graph traversal. The earlier
0/192 readout result used a different prompt/training protocol and remains
historical evidence of that recipe's failure, not a matched effect estimate.
Reasoning must be compared on fresh cases after freezing a working interface.
Direct lexical use of additional entity names requires an encoder extension.
External IDs can instead be mapped to these local aliases, with the same small
subgraph bound and without entity-specific biological meaning. Reading a stored
source label does not assess that source's quality.

Peak Metal allocation during this evaluation was 21.13 GB;
aggregate host pressure level was 1 at completion.
Other sessions were preserved. Frozen backbone hashes match before and after.
A development handoff briefly overlapped two own model loads and increased
swap; both were stopped. See `artifacts/readout-resource-incident.json`.
Subsequent runs enforce an exclusive model lease and deferred cancellation.

See [protocol and amendments](READOUT-REPAIR-PROTOCOL.md), the public raw
generations in `artifacts/readout-repair-test-v5`, and the source/checkpoints in
`artifacts/readout-curriculum-v5`. Reproduction commands:

```bash
.venv/bin/python scripts/train_readout_curriculum.py --output runs/readout-reproduce --seed 2303 --checkpoint artifacts/readout-curriculum-v4/step-0512.safetensors --field-curriculum --field-warmup 0 --prefix-noise 0.5 --updates 4096
.venv/bin/python scripts/evaluate_readout_repair.py --training runs/readout-reproduce --output runs/readout-reproduce-test
.venv/bin/python scripts/verify_artifacts.py
```

The archived `source` directories preserve each exact executed revision.
Regenerate the initial fresh-bank checkpoint from its archived training source,
or use the published BF16 warm-start checkpoint above. Online frozen-prefix
features are recomputed; there is no test-label feature cache.
