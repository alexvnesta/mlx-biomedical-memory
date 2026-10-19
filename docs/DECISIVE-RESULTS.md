# Learned memory versus trained text on unseen relationships

**Historical failed training recipe.** The subsequent [readout repair](READOUT-REPAIR-RESULTS.md) achieved 256/256 fresh complete records. The results below retain the original failed-access experiment unchanged. See the [new retention-gated protocol](COMPOSITION-PROTOCOL.md) for the reasoning follow-up.

The training recipe did not establish reliable memory access at the registered budget. This is a failed prerequisite for the proposed reasoning advantage. The held-out results below compare the actual trained reader with the strongest text reader; they do not establish that reliable memory is intrinsically unable to support reasoning.

Two training seeds, a pinned LongCat model, identical sample sequences and supervised path traces. Text has 5,914,624 trainable parameters; memory has 5,719,809 including its codec. Both receive every atomic fact. Longer paths in the main table retain the training bank size and record-ID vocabulary.

| Reader | Fresh-world labels | Fresh-world valid proofs | Depth 3–4 labels | Depth 3–4 valid proofs | Atomic exact |
|---|---:|---:|---:|---:|---:|
| Text + LoRA | 96/192 (50.0%) | 84/192 (43.8%) | 52/192 (27.1%) | 13/192 (6.8%) | 192/192 (100.0%) |
| Trained memory | 45/192 (23.4%) | 3/192 (1.6%) | 50/192 (26.0%) | 1/192 (0.5%) | 0/192 (0.0%) |
| Registered memory selection | 48/192 (25.0%) | 44/192 (22.9%) | 48/192 (25.0%) | 44/192 (22.9%) | 0/192 (0.0%) |
| Exact graph solver (oracle) | 100% | 100% | 100% | 100% | 100% |

Each neural stratum has 96 worlds evaluated by two fixed fitted readers. Atomic exactness covers 96 readout questions per seed across the 10- and 12-record banks. UNKNOWN uses the correct absence label and citations to every record; UP, DOWN and BOTH require executable paths. The interactive report separates these classes. The exact graph solver defines the synthetic gold labels. Its 100% is a rule-execution oracle, not an independent production-KG or ARAX evaluation; it has no LLM verbalization.

The registered memory selection may retain an untrained checkpoint. It is shown separately and must not be described as successfully trained. The trained-memory diagnostic requires a positive training step and uses only already-scored development results for selection.

## Path-present cases only

The registered all-world score above includes UNKNOWN. The preplanned class breakdown below uses UP, DOWN and BOTH, where the reader must generate an actual executable path. It prevents correct absence guesses from being mistaken for path reasoning.

| Reader | Fresh depth 1–2 paths | New depth 3–4 paths, 10 records |
|---|---:|---:|
| text | 73/144 (50.7%) | 10/144 (6.9%) |
| memory_trained | 3/144 (2.1%) | 0/144 (0.0%) |
| memory | 0/144 (0.0%) | 0/144 (0.0%) |
| Exact graph solver (oracle) | 72/72 (100%) | 72/72 (100%) |

### Actual paths by designed depth

Depth 1 requires no multi-hop composition. Depths 2–4 do. UNKNOWN cases are excluded from this table.

| Depth | Text + LoRA | Trained memory |
|---|---:|---:|
| 1 | 41/72 (56.9%) | 3/72 (4.2%) |
| 2 | 32/72 (44.4%) | 0/72 (0.0%) |
| 3 | 6/72 (8.3%) | 0/72 (0.0%) |
| 4 | 4/72 (5.6%) | 0/72 (0.0%) |

## Changes to evidence and bindings

| Reader | Edit pairs: both proofs valid | Permuted record IDs: both proofs valid | Reordered rows: both proofs valid |
|---|---:|---:|---:|
| text | 13/128 (10.2%) | 24/128 (18.8%) | 3/64 (4.7%) |
| memory_trained | 0/128 (0.0%) | 0/128 (0.0%) | 0/64 (0.0%) |
| memory | 0/128 (0.0%) | 26/128 (20.3%) | 14/64 (21.9%) |

On the same 128 worlds, trained-memory supported proofs change from 0 with the codec to 2 with it disabled.

## Fixed-length timing

Selected text and trained-memory readers from seed 1201, 32 generated tokens, EOS suppressed. Each condition has 24 requests. Warm requests reuse facts/instruction KV states and tokenize only the unprimed suffix. Setup costs include full prefix serialization, tokenization and prefill. Adapter changes and model loading are excluded. These are not timings to a correct answer. The Mac remained shared with other open sessions; small timing differences should not be treated as isolated-host performance.

| Condition | Median wall time | Median first token | Prompt tokens evaluated |
|---|---:|---:|---:|
| text-fresh | 0.589 s | 0.175 s | 424.5 / 424.5 |
| text-reused | 0.480 s | 0.083 s | 30.5 / 424.5 |
| memory-fresh | 0.532 s | 0.144 s | 263.5 / 263.5 |
| memory-reused | 0.470 s | 0.080 s | 30.5 / 263.5 |

Exact in-process traversal of these small complete graphs takes a median 3.2 microseconds per reasoning world, excluding network retrieval and LLM verbalization. A useful speed advantage requires comparable supported-answer quality.

## What the result addresses

This is a new engineered tuple codec plus LoRA, not an EngramEdit replication. Text facts pass through all 14 pretrained blocks; latent records enter before the last four. The local vocabulary is closed, with 16 node aliases. Data use fresh world namespaces and facts, not new biomedical terminology. The original 12-record structural stratum also introduces two untrained record IDs, so the separately frozen 10-record depth control is essential.

The training-bank causal audit, seed-level scores, field-level readouts, original stress stratum, every development candidate and every raw generation are available in the interactive report and public artifacts. All model runs are serial. Metal allocations are limited to 27 GB; aggregate host pressure and swap are retained without publishing other sessions or application names. Frozen backbone hashes are checked before and after training and evaluation.

Replacement of supplied records tests updateability without fitting new weights. It does not implement autonomous evidence ingestion, retention across tasks, calibrated biomedical confidence or large-graph scaling. Neither a positive nor a negative score here establishes rare-disease discovery or a comparison with OpenEvidence or open-web retrieval.

## Reproduce and inspect

- [Registered protocol](DECISIVE-PROTOCOL.md) and linked pre-test clarifications.
- [Reproduction instructions](DECISIVE-REPRODUCTION.md).
- [Design and interpretation](DECISIVE-DESIGN.md).
- [Full interactive report](../artifacts/decisive.html); download and open locally or serve the artifacts directory.
- [Machine-readable summary](../artifacts/decisive-summary.json).
- [Checksums and independent replay](../scripts/verify_decisive_artifacts.py).
