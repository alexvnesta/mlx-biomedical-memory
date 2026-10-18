# Experimental protocol and interpretation

## Scientific question

Can separately trained conditional latent memory support new compositions of
stored atomic relations, and does it improve on an identical frozen model given
the same relevant relations as text?

The initial implementation supplies the path to evaluate. It does not discover
an unmentioned intermediate entity. "Unseen relationship" means an untrained
composition of available premises, not unseen premises or a validated new
biological relationship.

## Conditions and scoring

1. **Base:** memory injection disabled, no premise text.
2. **Text facts:** memory disabled, exact relevant premises in the prompt.
3. **Memory:** learned rows activated by the supplied atomic phrases.
4. **Shuffled memory:** occupied addresses permuted; null addresses stay null.

The original LM's logits for the single-token options A and B are compared. Every
condition has the same options, rules, question, and scoring procedure. The text
condition has additional premise text by design. Its result is an oracle-retrieval
baseline, not an end-to-end retrieval benchmark. Reasoning-token generation,
retrieval latency, and explanation quality are not measured. A narrow classification
win must not be presented as general reasoning or biomedical superiority.

## Synthetic graph

Random identifiers label compound, gene, and process nodes. Every component has
two signed edges. Each split contains equal numbers of all four sign combinations.
An edge sign is an arbitrary stipulated value, not a biological association.
Atomic examples for all splits are fitted. Only training components' composed
labels enter optimization. Validation/test complete paths and components are
disjoint from training-composition supervision. Test labels are used only for
scoring, including the explicitly marked exploratory pilot runs.

## ARAX case study

Live TRAPI responses are retained with timestamp, full request, response SHA256,
schema/Biolink versions, provider status, and warnings. Disease identities are
resolved through ARAX. The focused monogenic queries use GLA (NCBIGene:2717),
GBA1 (NCBIGene:2629), and MECP2 (NCBIGene:4204), respectively. These are query
anchors, not causal claims inferred by the model. The initial broad queries also
remain local for comparison.

Only edges bound to returned query results enter the snapshot. Exact predicates,
directions, qualifiers, source roles, and publication attributes survive
normalization. The query's gene slot is screened for NCBIGene identifiers; broad
family concepts such as "Enzymes" are excluded from the benchmark. This is a
conservative operational screen, not comprehensive entity validation.

The memory benchmark uses predicate-level edge membership: subject, predicate,
object. It does not encode qualifier-specific mechanisms or source strength into
the learned values. The local evidence browser retains those details separately.
This intentional simplification must not be described as a complete biomedical
knowledge representation.

The graph labels are **recorded / unobserved**. Unobserved controls are endpoint
corruptions using existing identifiers of the same CURIE family, absent from the
union of retained bound edges. The shared gene remains unchanged, so positive
and negative supplied paths remain structurally connected. This is snapshot
membership, not positive/negative biological evidence. The model sees anonymous
entity aliases, exact predicate strings, and no source labels or case names.

Three copies of each positive path balance the three negative combinations. This
changes class weighting, not sample size. Reports must state unique question and
component counts; repeated positives are not independent discoveries. The three
diseases do not establish broad rare-disease generalization.

Provider warnings can indicate missing auxiliary support edges even when ARAX
reports success. Bound edges are usable for testing snapshot membership, while
scientific interpretation of their support can remain incomplete. Missing
publication metadata is not evidence of absence. Source records have not been
independently adjudicated in this prototype.

## Optimization and controls

The Qwen backbone is frozen, and complete tensor byte digests are compared before
and after fitting. Only the memory table, projection, and scalar gate are updated.
The disabled-memory wrapper is tested against the stock forward computation.
Freeze status alone does not imply unchanged behavior when memory is enabled.

Atomic and composed batches alternate. Same-length token batches avoid padding
changes. AdamW uses learning rate 0.003 and zero weight decay. The best validation
checkpoint is selected at fixed evaluation intervals; ties retain the first.
Memory and address manifests are saved independently of the backbone.

After the random-row pilots and seeds 23/42/71, the implementation was revised to
start memory rows at zero. The rationale is neutral initial behavior and avoiding
large arbitrary values in rarely updated rows. Seeds 101/202/303 use this fixed
initialization change; all preceding outcomes remain published. These are still
small exploratory experiments, not a preregistered confirmatory study.

The early 4-step execution smoke test, 64-dimensional pilot, and one-dimensional
bottleneck pilot are exploratory. Architecture changes made after those outcomes
must be described as such. Subsequent seeds must be retained regardless of result.
Every completed run is exported, including failures to improve. Source hashes,
dataset hashes, and exact dependency lock files make changes auditable.

## Gates before any whitepaper claim

- High atomic recall is necessary but insufficient for composition.
- Correct memories should outperform disabled and shuffled controls on held-out
  questions; paraphrases should retain the effect.
- Evidence of improvement over LLM+KG requires beating the supplied-text baseline,
  not merely beating the unaugmented model.
- Repeat across seeds and larger disjoint graphs. Use component-level uncertainty
  estimates; related paths and duplicate positive rows are not independent samples.
- Add LoRA with matched data/parameter budgets, generated-answer baselines,
  update/locality interventions, longer paths, and prospective graph holdouts.
- A biomedical claim additionally needs relation-specific expert review, primary
  source adjudication, exposure/context constraints, and appropriate external
  validation. Graph connectivity alone does not establish efficacy or safety.
