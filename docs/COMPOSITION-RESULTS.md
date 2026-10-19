# Composition after reliable memory access

Every selected reader retained at least 95% complete atomic readout on final banks. The path results therefore concern relationship queries in this particular interface. Development retention failures, excluded from final tests: memory-2402, joint-2401, joint-2402.

The separate [readout repair](READOUT-REPAIR-RESULTS.md) achieved 256/256 exact
fresh records, with value changes, bindings, order and masking controls. This
comparison uses new worlds, held-out TRACE endpoint pairs, a larger text adapter,
matched reasoning samples and budgets, two optimization seeds, and explicit
readout-retention eligibility. Full text and memory receive every atomic fact.

| Reader | Complete atomic records | One-hop valid paths | Two-hop valid paths | Three-hop valid paths | Four-hop valid paths |
|---|---:|---:|---:|---:|---:|
| Full text (2 eligible seeds) | 256/256 (100.0%) | 47/72 (65.3%) | 26/72 (36.1%) | 2/72 (2.8%) | 3/72 (4.2%) |
| Frozen memory encoder (1 eligible seed) | 124/128 (96.9%) | 1/36 (2.8%) | 0/36 (0.0%) | 0/36 (0.0%) | 0/36 (0.0%) |
| Joint memory adaptation (0 eligible seeds) | n/a | n/a | n/a | n/a | n/a |
| Exact graph solver (defines gold) | 100% | 100% | 100% | 100% | 100% |

These path columns exclude UNKNOWN: a valid absence guess is not a positive
path. A proof must execute through directed edges with matching species and
variant and correct sign product. Neural denominators include only eligible fitted
readers on the same worlds; they are descriptive counts, not independent
training-seed replications. The oracle defines the synthetic gold, not an
independent production-KG or ARAX benchmark.

## Per optimization seed

| Reader | Seed | Atomic exact | Fresh positive paths | Depth 3–4 positive paths | Selected update |
|---|---:|---:|---:|---:|---:|
| Full text | 2401 | 128/128 (100.0%) | 36/72 (50.0%) | 2/72 (2.8%) | 2048 |
| Full text | 2402 | 128/128 (100.0%) | 37/72 (51.4%) | 3/72 (4.2%) | 1088 |
| Frozen memory encoder | 2401 | 124/128 (96.9%) | 1/72 (1.4%) | 0/72 (0.0%) | 448 |
| Frozen memory encoder | 2402 | test closed; dev best 62/64 | not evaluated | not evaluated | ineligible |
| Joint memory adaptation | 2401 | test closed; dev best 61/64 | not evaluated | not evaluated | ineligible |
| Joint memory adaptation | 2402 | test closed; dev best 60/64 | not evaluated | not evaluated | ineligible |

## Same worlds and matched optimization seeds

These are the full positive-case primary strata, restricted only to common
eligible optimization seeds. They do not select cases by outcome or by support
readability. A failed retention fit remains excluded and explicitly reported.

| Comparison | Common seeds | Stratum | Memory valid proofs | Text valid proofs |
|---|---|---|---:|---:|
| Frozen memory encoder vs text | 2401 | fresh depth 1–2 | 1/72 (1.4%) | 36/72 (50.0%) |
| Frozen memory encoder vs text | 2401 | new depth 3–4 | 0/72 (0.0%) | 2/72 (2.8%) |
| Joint memory adaptation vs text | none | fresh depth 1–2 | n/a | n/a |
| Joint memory adaptation vs text | none | new depth 3–4 | n/a | n/a |

## Did the models have access to the required facts?

For the preselected support-probe worlds, the evaluator separately asked READ
for every record in a canonical support set. It never supplied those oracle
record IDs to the primary TRACE call. The following paired subset includes only
world/seed pairs where **both** readers reconstructed every required record
exactly. This is a diagnostic subset, not a replacement primary denominator.

| Paired comparison | Common support-readable cases | Memory valid proofs | Text valid proofs |
|---|---:|---:|---:|
| Frozen memory encoder vs text | 30 | 0/30 (0.0%) | 7/30 (23.3%) |
| Joint memory adaptation vs text | 0 | n/a | n/a |

## Combining values when the correct path is supplied

The oracle supplies path record IDs, never a sign or answer. Sweep every sign
assignment on those edges and flip an irrelevant edge independently. Inputs
stop before the next sign token. The table counts world/reader pairs where
**every** setting is correct, followed by overall exact sign-token counts. This
is not free path finding or complete JSON generation.
Each depth has eight new worlds per eligible fitted reader. A constant,
single-edge or all-bank product shortcut cannot pass every setting.

| Reader | One edge, all settings | Two edges, all settings | Three edges, all settings | Four edges, all settings |
|---|---:|---:|---:|---:|
| Full text | 16/16 worlds; 64/64 tokens | 5/16 worlds; 103/128 tokens | 0/16 worlds; 133/256 tokens | 0/16 worlds; 245/512 tokens |
| Frozen memory encoder | 8/8 worlds; 32/32 tokens | 0/8 worlds; 38/64 tokens | 0/8 worlds; 64/128 tokens | 0/8 worlds; 128/256 tokens |
| Joint memory adaptation | not evaluated | not evaluated | not evaluated | not evaluated |

Three- and four-edge chains were not reasoning-training depths. This diagnostic
separates retrieval/search from combining supplied records. Its masks and
all raw per-token decisions are in the public artifacts. Input-only prefix states contain
oracle path IDs but no output sign-target state. It does not replace the primary
free-proof test, and its sign algebra is not biological causal validation.

## Interpretation


On identical positive cases and the common optimization seed, memory produced 1/144 valid proofs and text 38/144. Both readers exactly reconstructed every required record in 30 probed worlds; their valid proofs there were 0 and 7. Those READ probes did not provide their IDs to the main TRACE call. Memory failed 30/30 proofs despite separately successful required-record READ probes.


With a supplied one-edge path, memory retrieved 32/32 signs correctly, versus 16/32 when masked. With two edges, it produced 38/64 correct sign tokens and 0/8 worlds with every setting correct. Supplying the path did not yield complete success on all two-edge worlds. This rules out path discovery as the sole problem; it does not isolate joint record access, representation alignment or learning the sign operation as the remaining cause.


Neither interface passed a complete three- or four-edge sign truth table. Those lengths were not reasoning-training depths. These descriptive counts concern this supplied-path diagnostic, rather than free path discovery or biological inference.


This prototype demonstrates an editable record-reading interface, with no demonstrated advantage in reasoning, quality-matched latency, scale or biomedical confidence over graph/text methods. The exact graph already executes these formal rules. Preserve the graph and evidence store; the learned interface remains experimental. The result is bounded to this encoder, late injection, adapter capacities and training recipe, with 3 failed fits excluded under the declared gate.


## Scope and costs

The atomic rehearsal pool uses the pair split from readout repair. All
self-relations fall in its development partition, so the retention gate also
tests that structural shift. Edges in TRACE training banks are unrestricted;
they are not all excluded by the atomic pair split. A failed gate is therefore
not evidence that every available graph fact was unreadable, or that another
retention curriculum could not work. It keeps that fitted reader unqualified
under this declared rule. The separate readout report documents the equality
diagnostic and the normal test's different pair distribution.
READ also uses the full graph-task instructions here, whereas the repair used
a focused READ prompt. These checks qualify the actual fitted interface used
for TRACE; they do not isolate forgetting from prompt-context transfer.

The repaired encoder adds readout-alignment training before this experiment.
Text updates 5,914,624 parameters; frozen memory updates 2,957,312 reader
parameters and retains its fixed encoder; joint memory updates 5,719,809.
Joint codec learning rate is 0.00001, reader rates are 0.00003. Both memory
encoders remain query-independent and receive only atomic record fields.

Memory is injected after ten frozen decoder blocks and has four blocks after
injection; explicit text is available from the input. This is a meaningful
processing-depth and representation difference. Results apply to this late-slot
prototype, not every learned-memory or Engram architecture. The world vocabulary
is small and anonymous; this is not rare-disease discovery or clinical validation.
No ranking against OpenEvidence, web recall or commercial evidence services was
measured. The exact graph is already sufficient for the stipulated formal rules.

Peak measured Metal allocation for these fits/evaluations was 21.67 GB.
Large processes were serialized with a kernel lease; tail training leaves more
headroom than full-model training. The earlier readout handoff incident is
retained in `artifacts/readout-resource-incident.json`.
The guard recorded 1 host-warning exit(s). Saved generations were retained; evaluation resumed with unchanged checkpoint/source hashes only after normal pressure returned. The warning's allocation cause is not attributed. Other sessions were not stopped.

See the [base protocol](COMPOSITION-PROTOCOL.md), [joint amendment](COMPOSITION-JOINT-AMENDMENT.md),
the [reproduction commands](COMPOSITION-REPRODUCTION.md),
and public raw outputs, checkpoint/source hashes, and frozen registrations in
`artifacts/composition-*`. Final test targets were never prefix-cached for
training. Facts, queries, generation strings and proof replay are public.
