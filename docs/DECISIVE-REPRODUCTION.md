# Reproducing the learned memory comparison

Use an Apple silicon Mac with adequate free unified memory and the pinned Python
packages in `pyproject.toml`. The measured host has 64 GiB; runtime receipts give
actual resource use. Download the pinned LongCat snapshot using the existing
setup instructions. Do not run a second large model concurrently.

The registered data are `artifacts/decisive-v1/dataset.json`. To regenerate them:

```bash
.venv/bin/python -c "from engram_poc.decisive_data import build; build('data/processed/decisive-v1/dataset.json')"
.venv/bin/python scripts/probe_decisive_runtime.py
.venv/bin/python scripts/cache_decisive_features.py
.venv/bin/python scripts/prepare_depth_control.py
.venv/bin/python scripts/run_decisive_experiment.py
.venv/bin/python scripts/run_decisive_graph_reference.py
.venv/bin/python scripts/run_decisive_depth_tests.py
.venv/bin/python scripts/run_decisive_trained_diagnostic.py
.venv/bin/python scripts/benchmark_decisive_latency.py
.venv/bin/python scripts/report_decisive.py
```

The driver runs serially, preserves each candidate evaluation, applies the
registered development gate, then evaluates both selected readers on all unseen
worlds and interventions. It supports restarting: existing completed work is
preserved, incomplete evaluations resume only against identical sources and
checkpoint hashes. An interrupted training run uses its saved optimizer and
FP32 weights. The primary evaluated checkpoints are BF16.

The feature-cache script refuses an existing directory without `--resume`.
Cache resumption requires the same data and generating source; start a new
output directory after any recipe change. Caches contain only causal
teacher-forced training/development states, with a public digest manifest.
No held-out answer is used to prepare a training feature.

To inspect the published generations without loading a model:

```bash
python3 -m http.server 8769 --directory artifacts
```

Open `http://127.0.0.1:8769/decisive.html`. All filtering is local.
The complete dataset, generated development selection scores, held-out prompts,
outputs, expected answers, executable-path scores, selected checkpoints and
aggregate resource receipts are public. BF16 selection candidates are also public. FP32 optimizer/resume checkpoints
are local intermediates; the checkpoint inventory records their hashes and
sizes, and archived source regenerates them. Frozen prefix feature tensors are
large local intermediates with public input/source/feature hashes.

To independently verify the public package on Linux or macOS, without MLX:

```bash
python3 scripts/verify_artifacts.py
python3 -m pytest tests/test_decisive_data.py -q
```

The artifact verifier replays scoring from the archived experiment semantics,
checks the data and source hashes, verifies each selection against development
scores, confirms that the published BF16 checkpoint is the one evaluated, and
checks fresh-world, order and binding inputs against the frozen dataset.

## Interpreting the study

The text reader has access to all fields and slightly more trainable capacity.
Both conditions use the same sample sequence, supervised targets and update
budget. The memory codec is query-independent and operates on atomic records;
no external graph solver supplies its generated answers. Exact graph traversal
is reported separately as a practical reference for these explicit formal rules.

LongCat uses its nonthinking chat mode in both conditions. Supervision explicitly
contains path traces; the model must generate them, and the independent verifier
checks them. This is not a comparison with a separately optimized reasoning-mode
model, frontier agent, web search system or clinical evidence service.

A failed development atomic-readout gate means the interface has not established
reliable memory access at the registered training budget. A negative held-out
comparison in that case cannot identify a failure of general reasoning in an
otherwise reliable memory reader. Report the failed prerequisite directly.
Conversely, successful readout without improved unseen paths would separate
factual access from composition. Faster fixed-token generation is useful only
at comparable supported-answer quality; text KV reuse is an essential control.

Text facts traverse all 14 pretrained blocks. Memory vectors enter after block
10, leaving four contextual blocks to interpret the interface. Both conditions
adapt those last four blocks, but the input representations have different
processing depth. This bounded frozen-prefix design does not test a fully
trained memory architecture. Its record-ID and entity vocabularies also cap the
world size; it cannot substantiate million-record scaling. Replacing atomic
records at inference tests updateability, not autonomous continuous learning.
