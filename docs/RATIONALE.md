# What would make latent memory useful?

The motivating gap is reliable reuse of changing knowledge without retraining an
entire language model. A useful system must retrieve the right qualified claim,
apply it to a new situation, preserve its source and scope, and respond correctly
when evidence changes. Compact storage alone does not establish any of these.

## What the alternatives already provide

| Approach | Useful capability | Remaining problem |
|---|---|---|
| Versioned knowledge graph | Explicit relationships, qualifiers, source links, cheap edits and deletion | Entity alignment, query planning and meaningful inference still need machinery |
| Text or hybrid RAG | Retrieve passages and update the index without backbone training | Retrieval can omit a necessary premise; fluent synthesis can exceed the evidence |
| LLM graph agent | Choose queries and combine returned records | Tool planning, stopping, context matching and proof completeness can fail |
| Scoped latent residual | Reuse a small fitted value or rule through an explicit address | Editing cost, interference, access reliability, rule application and provenance remain to be demonstrated |

The scoped prototype leaves qualified facts and provenance outside the language
model. Its rule bank could become a cache of frequently reused instructions. The
graph is still needed to supply current evidence. This is a hybrid hypothesis;
replacing an editable graph with vectors would discard valuable structure.

## A useful result must clear several gates

1. Improve factual access over the native subject-addressed baseline without
   hiding context selection inside an answer-producing router.
2. Apply a stored rule to new entities and evidence, with valid cited support.
3. Beat strong text-rule and graph-agent controls on quality or cost at comparable
   quality. Shorter prompts with worse answers do not establish efficiency.
4. Support reliable corrections, deletion and rollback, then measure interference
   across many edits and realistic address distributions.
5. Preserve evidence lineage and abstain when the required information is absent.

The [first comparison](COMPARISON.md) and [richer experiment](RELATIONAL-RESULTS.md)
are exploratory probes of these gates. They use bounded corpora and correlated
synthetic templates, not a validated rare-disease discovery benchmark. Model,
source and checkpoint hashes make individual outcomes auditable; they do not
make the benchmark representative of clinical work.

## Continuous learning and confidence

An explicitly fitted external memory is an update mechanism. It becomes a
continuous-learning system only after adding an evidence intake policy,
qualification of conflicting claims, update selection, regression monitoring,
versioning and rollback. The current code does not autonomously learn new facts
or new scientific rules from a stream.

An address hit is not confidence. A model can read the selected vector and still
generate the wrong value or misuse a rule. Formal proof checks provide a narrow
kind of assurance about supplied records under declared semantics. Biological
confidence additionally requires appropriate evidence, experimental scope,
study quality and uncertainty; distinct provider IDs are not independent studies.

## Practical scale and a meaningful use case

The strongest candidate use case is repeated, narrow queries over frequently
reused qualified knowledge or instructions, where a measured saving amortizes
the fitting and maintenance cost. A break-even estimate requires comparable
answer quality: total fitting and maintenance seconds divided by per-query
seconds saved. If quality falls or no speed advantage appears, this estimate is
not a justification for deployment.

Small residual files exclude the large backbone, graph, text corpus and retrieval
index. Fast dictionary addressing does not establish the quality of a learned
router or the stability of a million edits. This project has not validated those
scales. Prefix caching, explicit rule libraries and ordinary graph queries remain
important practical comparators before investing in a latent rule cache.

OpenEvidence and web search answer different, open-ended evidence questions.
They are not scored on these private synthetic worlds. A fair biomedical study
would freeze accessible source material and questions, audit source-level support
with domain experts, and compare systems on the same evidence and review task.
No superiority over either is claimed here.

Related primary work: [EngramEdit](https://arxiv.org/abs/2610.10533),
[ProofWriter](https://arxiv.org/abs/2012.13048), and
[Graph-CoT](https://arxiv.org/abs/2404.07103).
