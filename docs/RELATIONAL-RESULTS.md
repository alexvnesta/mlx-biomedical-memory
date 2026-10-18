# Richer relationships: measured results

An exploratory MLX comparison of scoped factual addresses and reusable latent rule definitions. These custom cases do not constitute rare-disease discovery or clinical validation. The earlier [comparison](COMPARISON.md) remains unchanged. [Frozen protocol](RELATIONAL-PROTOCOL.md).

**Measured conclusion.** With complete gold evidence, scoped rules produced 20/60 supported answers, versus 30/60 with text rules and 16/60 with rules disabled. For model-planned graph queries, scoped rules produced 7/60, versus 14/60 with text rules. This run does not demonstrate an advantage over supplying the rule as text.

The family table matters: scoped memory has some gains over disabled rules, but these do not establish reliable general rule application. In particular, the stipulated signed-path cases expose failures even with complete evidence and text rules. A faster completion at lower answer quality cannot establish practical efficiency. See [rationale and decision criteria](RATIONALE.md) for the remaining utility, confidence and scaling questions.

Fitted 24 scoped values and six generic rules in 905.0 seconds. Peak Metal allocation: fitting 39.23 GB; evaluation 39.81 GB. All base-model parameter digests match before and after fitting and inference.

The scoped checkpoint stores 30 non-null rows in 381,027 bytes; the jointly allocated n-gram checkpoint stores 18 non-null keys in 258,274 bytes. Values are rounded to BF16 and stored in FP32 containers; small checkpoint size excludes base weights and the external evidence index.

The scoped interface explicitly routes `(entity, relation, context)` to a residual at the entity token. It is an engineered adapter, not native n-gram addressing or a faithful EngramEdit reproduction. The same target fits supply both memory conditions, but scoped rows avoid joint allocation/regularization and use a different number of parameters. This is not a parameter-matched test isolating addressing alone.

## Scoped recall

| Condition | Atomic canonical (24) | Atomic alternate (24) | Atomic mean NLL | Rule exact (6) | Rule target prefix (6) | Rule mean NLL |
|---|---|---|---|---|---|---|
| Native model | 0/24 (0.0%) | 0/24 (0.0%) | 4.853 | 0/6 (0.0%) | 0/6 (0.0%) | 4.609 |
| Subject n-gram memory | 0/24 (0.0%) | 0/24 (0.0%) | 2.006 | 0/6 (0.0%) | 0/6 (0.0%) | 2.742 |
| Scoped tuple memory | 3/24 (12.5%) | 2/24 (8.3%) | 0.085 | 0/6 (0.0%) | 3/6 (50.0%) | 0.614 |
| Shuffled scoped memory | 1/24 (4.2%) | 0/24 (0.0%) | 5.974 | 0/6 (0.0%) | 0/6 (0.0%) | 4.874 |
| Text value supplied | 24/24 (100.0%) | 24/24 (100.0%) | 0.051 | 4/6 (66.7%) | 4/6 (66.7%) | 0.079 |

Atomic queries repeat fitted lookup tuples in two formats; they are an access/interference diagnostic, not unseen factual knowledge. Rule-prefix scoring requires the complete stored definition at the start of a completion. NLL is teacher-forced; none of these are reasoning scores. The deranged atomic control has 2/24 coincidentally identical target values because assigned opaque values can repeat.

## Reasoning and proof support

| Condition | Synthetic answer (48) | Synthetic proof (48) | ARAX answer (12) | ARAX proof (12) |
|---|---|---|---|---|
| Gold evidence + text rule | 27/48 (56.2%) | 27/48 (56.2%) | 3/12 (25.0%) | 3/12 (25.0%) |
| Gold evidence + scoped rule | 20/48 (41.7%) | 19/48 (39.6%) | 1/12 (8.3%) | 1/12 (8.3%) |
| Gold evidence + n-gram rule | 11/48 (22.9%) | 9/48 (18.8%) | 1/12 (8.3%) | 1/12 (8.3%) |
| Gold evidence + shuffled rule | 17/48 (35.4%) | 14/48 (29.2%) | 1/12 (8.3%) | 1/12 (8.3%) |
| Gold evidence, rule disabled | 18/48 (37.5%) | 15/48 (31.2%) | 1/12 (8.3%) | 1/12 (8.3%) |
| BM25 RAG + text rule | 24/48 (50.0%) | 14/48 (29.2%) | 3/12 (25.0%) | 3/12 (25.0%) |
| Hybrid RAG + text rule | 17/48 (35.4%) | 14/48 (29.2%) | 3/12 (25.0%) | 3/12 (25.0%) |
| Graph agent + text rule | 18/48 (37.5%) | 12/48 (25.0%) | 4/12 (33.3%) | 2/12 (16.7%) |
| Graph agent + scoped rule | 12/48 (25.0%) | 5/48 (10.4%) | 2/12 (16.7%) | 2/12 (16.7%) |
| Graph agent, rule disabled | 13/48 (27.1%) | 7/48 (14.6%) | 2/12 (16.7%) | 2/12 (16.7%) |

Gold evidence supplies the complete local view, not a model-discovered path. All conditions receive the same explicit rule address. Only rule text versus latent storage differs in the gold comparisons. Synthetic entities and entity-specific proofs never occur as fitting targets. Each family has eight correlated template instances. ARAX uses 12 queries over three bounded 12-edge views, retaining annotation scope and direction; it does not infer biology by multiplying graph signs. All 36 blinded ARAX records were separately checked against retained original edge hashes, including subject/object alias bijections, direction, qualifiers and provider roles.

## Retrieval and planning

| Condition | Answer (60) | Proof (60) | Median wall, s | Mean model calls | Mean lookups | Observed s per valid proof |
|---|---|---|---|---|---|---|
| Gold evidence + text rule | 30/60 (50.0%) | 30/60 (50.0%) | 0.534 | 1.00 | 0.00 | 1.13 |
| Graph agent + text rule | 22/60 (36.7%) | 14/60 (23.3%) | 1.046 | 1.98 | 1.15 | 4.34 |
| Graph agent + scoped rule | 14/60 (23.3%) | 7/60 (11.7%) | 0.989 | 2.00 | 1.15 | 9.05 |
| Graph agent, rule disabled | 15/60 (25.0%) | 9/60 (15.0%) | 1.022 | 2.00 | 1.17 | 7.15 |
| BM25 RAG + text rule | 27/60 (45.0%) | 17/60 (28.3%) | 0.588 | 1.00 | 0.00 | 2.16 |
| Hybrid RAG + text rule | 20/60 (33.3%) | 17/60 (28.3%) | 0.604 | 1.00 | 0.00 | 2.20 |

Graph agents choose their own entities, relations, directions and context filters. No gold start entity or hop count is passed separately. The shared budget is four model calls and six lookups, at most two per call and 16 records per lookup. BM25 and pinned BGE hybrid retrieve top-12 records globally; online CPU query encoding is included. Observed seconds per valid proof includes time spent on failed requests and is a cohort efficiency measure, not a clinical time-to-answer guarantee. Model loading, fitting and document embedding preparation are excluded. Variable output lengths affect ordinary wall times.

## Paired proof comparison

| Candidate | Reference | Candidate only | Reference only | Both valid | Neither valid |
|---|---|---|---|---|---|
| Gold evidence + scoped rule | Gold evidence + text rule | 6 | 16 | 14 | 24 |
| Gold evidence + scoped rule | Gold evidence, rule disabled | 7 | 3 | 13 | 37 |
| Gold evidence + n-gram rule | Gold evidence, rule disabled | 1 | 7 | 9 | 43 |
| Graph agent + scoped rule | Graph agent + text rule | 0 | 7 | 7 | 46 |
| Graph agent + scoped rule | Graph agent, rule disabled | 0 | 2 | 7 | 51 |

These are matched outcomes on the same 60 questions, including the 12 ARAX audits. They describe where one condition succeeds while another fails; no significance or population-level effect is inferred from correlated templates.

## Counterfactual evidence

| Condition | Updated answer (6) | Updated proof (6) | Both paired answers correct |
|---|---|---|---|
| Gold evidence + text rule | 3/6 (50.0%) | 3/6 (50.0%) | 2/6 |
| Gold evidence + scoped rule | 1/6 (16.7%) | 1/6 (16.7%) | 1/6 |
| Graph agent + text rule | 2/6 (33.3%) | 1/6 (16.7%) | 1/6 |
| Graph agent + scoped rule | 1/6 (16.7%) | 0/6 (0.0%) | 0/6 |
| Graph agent, rule disabled | 2/6 (33.3%) | 1/6 (16.7%) | 1/6 |

Six paired interventions change supplied records with rules and memory fixed. This measures responsiveness to changed evidence, not automatic learning or neural editing. Paired cases are not additional independent worlds.

## Fixed 32-token latency

| Condition | Requests | Prompt tokens | p50 seconds | p95 seconds |
|---|---|---|---|---|
| Gold evidence + text rule | 24 | 425 | 0.566 | 0.624 |
| Gold evidence + scoped rule | 24 | 400 | 0.546 | 0.644 |
| Gold evidence + n-gram rule | 24 | 400 | 0.591 | 0.650 |

Greedy completions are forced to exactly 32 tokens by suppressing EOS. There are 12 queries, two repetitions and randomized condition order. Model/OS caches are warm, but no prefix cache is reused. Prompt construction, addressing and tokenization are included; output quality is not matched.

## Failure audit

| Condition | Correct label, insufficient proof | Unretrieved citations | Missing final answer | Tool errors |
|---|---|---|---|---|
| Gold evidence + text rule | 0 | 0 | 0 | 0 |
| Gold evidence + scoped rule | 1 | 0 | 0 | 0 |
| Gold evidence + n-gram rule | 2 | 0 | 0 | 0 |
| Gold evidence + shuffled rule | 3 | 0 | 0 | 0 |
| Gold evidence, rule disabled | 3 | 0 | 0 | 0 |
| BM25 RAG + text rule | 10 | 0 | 0 | 0 |
| Hybrid RAG + text rule | 3 | 0 | 0 | 0 |
| Graph agent + text rule | 8 | 0 | 0 | 0 |
| Graph agent + scoped rule | 7 | 0 | 0 | 0 |
| Graph agent, rule disabled | 6 | 0 | 0 | 0 |

## Synthetic families

| Family | Condition | Answer (8) | Proof (8) |
|---|---|---|---|
| join | Gold evidence + text rule | 8/8 (100.0%) | 8/8 (100.0%) |
| join | Gold evidence + scoped rule | 4/8 (50.0%) | 3/8 (37.5%) |
| join | Gold evidence + n-gram rule | 4/8 (50.0%) | 2/8 (25.0%) |
| join | Gold evidence + shuffled rule | 6/8 (75.0%) | 6/8 (75.0%) |
| join | Gold evidence, rule disabled | 5/8 (62.5%) | 4/8 (50.0%) |
| join | BM25 RAG + text rule | 5/8 (62.5%) | 1/8 (12.5%) |
| join | Hybrid RAG + text rule | 4/8 (50.0%) | 3/8 (37.5%) |
| join | Graph agent + text rule | 4/8 (50.0%) | 0/8 (0.0%) |
| join | Graph agent + scoped rule | 4/8 (50.0%) | 0/8 (0.0%) |
| join | Graph agent, rule disabled | 4/8 (50.0%) | 0/8 (0.0%) |
| context | Gold evidence + text rule | 5/8 (62.5%) | 5/8 (62.5%) |
| context | Gold evidence + scoped rule | 2/8 (25.0%) | 2/8 (25.0%) |
| context | Gold evidence + n-gram rule | 0/8 (0.0%) | 0/8 (0.0%) |
| context | Gold evidence + shuffled rule | 2/8 (25.0%) | 2/8 (25.0%) |
| context | Gold evidence, rule disabled | 3/8 (37.5%) | 3/8 (37.5%) |
| context | BM25 RAG + text rule | 3/8 (37.5%) | 3/8 (37.5%) |
| context | Hybrid RAG + text rule | 3/8 (37.5%) | 3/8 (37.5%) |
| context | Graph agent + text rule | 4/8 (50.0%) | 4/8 (50.0%) |
| context | Graph agent + scoped rule | 0/8 (0.0%) | 0/8 (0.0%) |
| context | Graph agent, rule disabled | 0/8 (0.0%) | 0/8 (0.0%) |
| sign | Gold evidence + text rule | 0/8 (0.0%) | 0/8 (0.0%) |
| sign | Gold evidence + scoped rule | 0/8 (0.0%) | 0/8 (0.0%) |
| sign | Gold evidence + n-gram rule | 0/8 (0.0%) | 0/8 (0.0%) |
| sign | Gold evidence + shuffled rule | 0/8 (0.0%) | 0/8 (0.0%) |
| sign | Gold evidence, rule disabled | 0/8 (0.0%) | 0/8 (0.0%) |
| sign | BM25 RAG + text rule | 2/8 (25.0%) | 0/8 (0.0%) |
| sign | Hybrid RAG + text rule | 1/8 (12.5%) | 0/8 (0.0%) |
| sign | Graph agent + text rule | 2/8 (25.0%) | 0/8 (0.0%) |
| sign | Graph agent + scoped rule | 3/8 (37.5%) | 0/8 (0.0%) |
| sign | Graph agent, rule disabled | 2/8 (25.0%) | 0/8 (0.0%) |
| provenance | Gold evidence + text rule | 2/8 (25.0%) | 2/8 (25.0%) |
| provenance | Gold evidence + scoped rule | 5/8 (62.5%) | 5/8 (62.5%) |
| provenance | Gold evidence + n-gram rule | 2/8 (25.0%) | 2/8 (25.0%) |
| provenance | Gold evidence + shuffled rule | 2/8 (25.0%) | 2/8 (25.0%) |
| provenance | Gold evidence, rule disabled | 2/8 (25.0%) | 2/8 (25.0%) |
| provenance | BM25 RAG + text rule | 2/8 (25.0%) | 2/8 (25.0%) |
| provenance | Hybrid RAG + text rule | 2/8 (25.0%) | 2/8 (25.0%) |
| provenance | Graph agent + text rule | 4/8 (50.0%) | 4/8 (50.0%) |
| provenance | Graph agent + scoped rule | 2/8 (25.0%) | 2/8 (25.0%) |
| provenance | Graph agent, rule disabled | 3/8 (37.5%) | 3/8 (37.5%) |
| conflict | Gold evidence + text rule | 8/8 (100.0%) | 8/8 (100.0%) |
| conflict | Gold evidence + scoped rule | 5/8 (62.5%) | 5/8 (62.5%) |
| conflict | Gold evidence + n-gram rule | 5/8 (62.5%) | 5/8 (62.5%) |
| conflict | Gold evidence + shuffled rule | 4/8 (50.0%) | 4/8 (50.0%) |
| conflict | Gold evidence, rule disabled | 4/8 (50.0%) | 4/8 (50.0%) |
| conflict | BM25 RAG + text rule | 8/8 (100.0%) | 8/8 (100.0%) |
| conflict | Hybrid RAG + text rule | 6/8 (75.0%) | 6/8 (75.0%) |
| conflict | Graph agent + text rule | 4/8 (50.0%) | 4/8 (50.0%) |
| conflict | Graph agent + scoped rule | 3/8 (37.5%) | 3/8 (37.5%) |
| conflict | Graph agent, rule disabled | 4/8 (50.0%) | 4/8 (50.0%) |
| abduction | Gold evidence + text rule | 4/8 (50.0%) | 4/8 (50.0%) |
| abduction | Gold evidence + scoped rule | 4/8 (50.0%) | 4/8 (50.0%) |
| abduction | Gold evidence + n-gram rule | 0/8 (0.0%) | 0/8 (0.0%) |
| abduction | Gold evidence + shuffled rule | 3/8 (37.5%) | 0/8 (0.0%) |
| abduction | Gold evidence, rule disabled | 4/8 (50.0%) | 2/8 (25.0%) |
| abduction | BM25 RAG + text rule | 4/8 (50.0%) | 0/8 (0.0%) |
| abduction | Hybrid RAG + text rule | 1/8 (12.5%) | 0/8 (0.0%) |
| abduction | Graph agent + text rule | 0/8 (0.0%) | 0/8 (0.0%) |
| abduction | Graph agent + scoped rule | 0/8 (0.0%) | 0/8 (0.0%) |
| abduction | Graph agent, rule disabled | 0/8 (0.0%) | 0/8 (0.0%) |

## Post-hoc fitted-context diagnostic

| Suite | Format | Condition | Literal recall | Mean NLL |
|---|---|---|---|---|
| atomic | fitted_bare | Native model | 0/24 (0.0%) | 4.675 |
| atomic | fitted_bare | Subject n-gram memory | 0/24 (0.0%) | 2.001 |
| atomic | fitted_bare | Scoped tuple memory | 2/24 (8.3%) | 0.002 |
| atomic | fitted_reference | Native model | 0/24 (0.0%) | 4.520 |
| atomic | fitted_reference | Subject n-gram memory | 0/24 (0.0%) | 1.950 |
| atomic | fitted_reference | Scoped tuple memory | 3/24 (12.5%) | 0.001 |
| motif | fitted_bare | Native model | 0/6 (0.0%) | 4.378 |
| motif | fitted_bare | Subject n-gram memory | 0/6 (0.0%) | 2.558 |
| motif | fitted_bare | Scoped tuple memory | 1/6 (16.7%) | 0.560 |
| motif | fitted_reference | Native model | 0/6 (0.0%) | 4.889 |
| motif | fitted_reference | Subject n-gram memory | 0/6 (0.0%) | 2.721 |
| motif | fitted_reference | Scoped tuple memory | 0/6 (0.0%) | 0.584 |

This additional diagnostic was designed after observing weak atomic recall under the benchmark instruction prefix. It replays both exact fitting prefix contexts, uses the same fixed checkpoint, and does not replace any primary outcome. For every replay, the loaded scoped bank produces bit-identical full-sequence logits to the fitting-time probe with the same BF16-rounded delta. This checks injection correctness and distinguishes fitted-context access from transfer; it is not an additional reasoning benchmark.

## Post-hoc value-boundary control

| Format | Fitting target | Exact value (4) | Literal prefix only (4) |
|---|---|---|---|
| fitted_bare | original | 1/4 (25.0%) | 4/4 (100.0%) |
| fitted_bare | newline | 4/4 (100.0%) | 4/4 (100.0%) |
| fitted_reference | original | 1/4 (25.0%) | 4/4 (100.0%) |
| fitted_reference | newline | 4/4 (100.0%) | 4/4 (100.0%) |
| instruction | original | 1/4 (25.0%) | 4/4 (100.0%) |
| instruction | newline | 4/4 (100.0%) | 4/4 (100.0%) |
| question | original | 2/4 (50.0%) | 4/4 (100.0%) |
| question | newline | 3/4 (75.0%) | 4/4 (100.0%) |

The first four atomic records, selected by frozen dataset order rather than failure status, were separately refitted with a newline after each target. The optimizer, 25-step budget, templates and two training prefix contexts are unchanged. Original checkpoints and all primary outcomes remain intact. The original objective supervises target tokens without a termination token: a low loss for `G1` can coexist with generating `G14D`. Literal-prefix scoring deliberately ignores the following character and is not factual recall. This four-record control is exploratory, uses the same underlying records repeatedly, and does not establish reasoning or general update quality.

## Development interface checks

| Attempt | Condition | Answer (6) | Proof (6) |
|---|---|---|---|
| relational-dev-v1 | Gold evidence + text rule | 5/6 (83.3%) | 5/6 (83.3%) |
| relational-dev-v1 | Graph agent + text rule | 0/6 (0.0%) | 0/6 (0.0%) |
| relational-dev-v2 | Gold evidence + text rule | 5/6 (83.3%) | 5/6 (83.3%) |
| relational-dev-v2 | Graph agent + text rule | 0/6 (0.0%) | 0/6 (0.0%) |
| relational-dev-v3 | Gold evidence + text rule | 5/6 (83.3%) | 5/6 (83.3%) |
| relational-dev-v3 | Graph agent + text rule | 4/6 (66.7%) | 3/6 (50.0%) |

The same six separate development worlds were reused for interface diagnosis only. Attempts 1 and 2 used raw continuation prompts; attempt 3 uses the pinned native chat template with thinking disabled. All held-out reasoning and latency conditions use the final chat format. These development counts are not pooled with test outcomes.

## Interpretation limits

A valid proof means sufficient retrieved record IDs under stipulated executable rules, not a validated biological mechanism. Unknown/none can be certified by complete relevant lookups or a complete inspected core; an arbitrary empty subset is insufficient. Complete certificates are generated by the graph backend, never accepted as unsupported model claims. A model can have a correct label and an invalid evidence packet; those are counted separately.

The latent bank stores generic rule texts, not an autonomously learned theory of biology. Good recall need not transfer to rule application. Explicit addressing avoids one interference source but does not learn semantic retrieval or guarantee calibrated confidence. This is one frozen 4-bit backbone under greedy, thinking-disabled generation with a 192-token call budget; it does not establish the limits of other models, reasoning modes or all latent-memory methods. Different primary-provider IDs do not prove independent experiments. Tissue-specific reasoning, clinical evidence adjudication, prospective discovery and large-scale editing are not established. OpenEvidence and web search are not scored on private synthetic rules. The host is a 64 GiB M5 Max Mac; reported Metal peaks are not total system memory. The busy host had approximately 12 GiB swap in use around evaluation, so these are local observations rather than isolated hardware benchmarks.

## Reproduce

```bash
uv sync --extra dev --extra longcat
uv run --extra longcat python scripts/prepare_longcat.py --download-model
uv run --extra longcat python scripts/prepare_relational.py --replay --cache-encoder
uv run --extra longcat python scripts/fit_relational_memory.py
uv run --extra longcat python scripts/run_relational.py --dev-only --output runs/relational-dev-v3
uv run --extra longcat python scripts/run_relational.py
uv run --extra longcat python scripts/diagnose_scoped_transfer.py
uv run --extra longcat python scripts/diagnose_value_boundary.py
# Reuse the published audit. Omit --replay to recheck retained raw snapshots:
uv run python scripts/audit_relational_arax.py --replay
uv run python scripts/report_relational.py
uv run python scripts/verify_artifacts.py
```

The commands assume a fresh checkout with no local run outputs. Preserve completed runs when repeating; diagnostic/report defaults use the published version-1 paths. The current runner reproduces the final chat interface; earlier development interfaces are retained in their source snapshots. Inference resume is restricted to identical source, data and checkpoint hashes. Raw prompts, generated actions, audited tool parameters, citations, target-fit histories, checkpoints and source snapshots are public. Raw ARAX composite graphs and original entity bindings remain local.

Sources: [EngramEdit](https://arxiv.org/abs/2610.10533), [ProofWriter](https://arxiv.org/abs/2012.13048), [Graph-CoT](https://arxiv.org/abs/2404.07103). This is not a reproduction of their published benchmark scores.
