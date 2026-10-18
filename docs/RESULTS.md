# Measured results

All completed local runs are retained below. These are small exploratory experiments,
not evidence of a rare-disease discovery or general superiority over LLM+KG.

| Run | Base | Supplied text facts | Latent memory | Shuffled | Atomic recall with memory | Paraphrase |
|---|---:|---:|---:|---:|---:|---:|
| [arax-seed17](../artifacts/arax-seed17/results.json) | 50.0% | 100.0% | 50.0% | 52.8% | 61.5% | 50.0% |
| [pilot-bottleneck-seed17](../artifacts/pilot-bottleneck-seed17/results.json) | 50.0% | 75.0% | 75.0% | 50.0% | 62.5% | 50.0% |
| [pilot-mcq-v2-seed17](../artifacts/pilot-mcq-v2-seed17/results.json) | 50.0% | 75.0% | 50.0% | 37.5% | 75.0% | 37.5% |
| [replication-seed23](../artifacts/replication-seed23/results.json) | 50.0% | 75.0% | 62.5% | 43.8% | 50.0% | 37.5% |
| [replication-seed42](../artifacts/replication-seed42/results.json) | 50.0% | 75.0% | 50.0% | 56.2% | 56.2% | 43.8% |
| [replication-seed71](../artifacts/replication-seed71/results.json) | 50.0% | 75.0% | 43.8% | 62.5% | 50.0% | 37.5% |
| [smoke-seed17](../artifacts/smoke-seed17/results.json) | 50.0% | 50.0% | 50.0% | 50.0% | 50.0% | 50.0% |
| [zero-init-seed101](../artifacts/zero-init-seed101/results.json) | 50.0% | 75.0% | 50.0% | 56.2% | 50.0% | 68.8% |
| [zero-init-seed202](../artifacts/zero-init-seed202/results.json) | 50.0% | 75.0% | 43.8% | 31.2% | 75.0% | 50.0% |
| [zero-init-seed303](../artifacts/zero-init-seed303/results.json) | 50.0% | 75.0% | 50.0% | 43.8% | 50.0% | 50.0% |

## Interpretation

The initial pilots used randomly initialized memory rows. Low held-out atomic recall
limits conclusions about reasoning: a model cannot reliably compose premises that its
memory has not learned. A small composition score above the disabled control is not
sufficient evidence of systematic generalization. The supplied-text comparator is
stronger than end-to-end retrieval because it receives the exact required premises.

The first smoke run used a plain completion probe and four optimizer steps. It is an
execution check, not an architectural comparison. Later runs use chat-formatted A/B
probes. Early pilots projected only the final hidden state; the corrected wrapper
preserves the stock projection shape to make memory-disabled computation identical.

The synthetic tests have 16 held-out components unless the run is explicitly a smoke
test. ARAX rows include repeated positives for class weighting: count unique questions
and components, not rows, when assessing uncertainty. Run configurations and raw
per-example predictions are linked above. The code and data hashes are recorded in
later run configurations. Early pilots without a source hash remain exploratory.

See [protocol](PROTOCOL.md) for what the experiment does and does not establish.

## Fresh-seed summary

The three fixed-recipe zero-initialization runs have mean held-out composition accuracy 47.9% with latent memory, versus 75.0% with supplied text facts. This does not demonstrate a reasoning advantage. Each synthetic run has only 16 held-out components.

Prompt lengths are recorded separately in [token accounting](../artifacts/token-accounting.json). Fewer prompt tokens alone are not an efficiency benefit when answer accuracy is lower. No latency or amortized-training-cost advantage has been established.

## ARAX-derived snapshot task

The `arax-seed17` run has 72 weighted test rows, 46 unique supplied edge-pair questions, and 12 held-out Rett-syndrome path components. Latent-memory membership composition is 50.0%, versus 100.0% with exact text facts. Atomic recall with memory is 61.5%.

This is a conjunction of recorded/unobserved predicate-level edge memberships, not sign inference, hidden-path discovery, a treatment prediction, or validation of a new biomedical relationship. Weak atomic recall limits conclusions about a better memory interface.
