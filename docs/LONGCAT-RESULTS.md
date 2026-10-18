# LongCat MLX replay: measured results

Exploratory replay of the authors' MQuAKE editing state on 4-bit MLX
LongCat-Flash-Lite. This is not new biomedical training or an exact BF16
paper reproduction. See [protocol and limitations](LONGCAT.md).

The six-case smoke test used one question per case. The later 30-case replay
uses all three question variants and includes those initial six cases. It was
expanded after seeing the smoke result, without changing the checkpoint or
generation recipe. The overlapping runs are not independent replications.

| Run | Style | Condition | Updated-answer question accuracy | Cases with any updated answer |
|---|---|---|---:|---:|
| [longcat-mquake-30](../artifacts/longcat-mquake-30/summary.json) | standard | base | 3.3% (90 questions) | 3.3% |
| [longcat-mquake-30](../artifacts/longcat-mquake-30/summary.json) | standard | released_memory | 14.4% (90 questions) | 16.7% |
| [longcat-mquake-30](../artifacts/longcat-mquake-30/summary.json) | standard | text_edits | 82.2% (90 questions) | 86.7% |
| [longcat-mquake-30](../artifacts/longcat-mquake-30/summary.json) | cot | base | 1.1% (90 questions) | 3.3% |
| [longcat-mquake-30](../artifacts/longcat-mquake-30/summary.json) | cot | released_memory | 8.9% (90 questions) | 26.7% |
| [longcat-mquake-30](../artifacts/longcat-mquake-30/summary.json) | cot | text_edits | 80.0% (90 questions) | 90.0% |
| [longcat-mquake-smoke-v4](../artifacts/longcat-mquake-smoke-v4/summary.json) | standard | base | 0.0% (6 questions) | 0.0% |
| [longcat-mquake-smoke-v4](../artifacts/longcat-mquake-smoke-v4/summary.json) | standard | released_memory | 16.7% (6 questions) | 16.7% |
| [longcat-mquake-smoke-v4](../artifacts/longcat-mquake-smoke-v4/summary.json) | standard | text_edits | 100.0% (6 questions) | 100.0% |
| [longcat-mquake-smoke-v4](../artifacts/longcat-mquake-smoke-v4/summary.json) | cot | base | 0.0% (6 questions) | 0.0% |
| [longcat-mquake-smoke-v4](../artifacts/longcat-mquake-smoke-v4/summary.json) | cot | released_memory | 16.7% (6 questions) | 16.7% |
| [longcat-mquake-smoke-v4](../artifacts/longcat-mquake-smoke-v4/summary.json) | cot | text_edits | 100.0% (6 questions) | 100.0% |

## Validation and interpretation

The disabled wrapper matches tested native-model logits exactly. Native and
edited embeddings match exactly between full and chunked inputs. The actual
published residuals also match the pinned upstream PyTorch wrapper exactly
in BF16 on a 49-token test with identical synthetic base outputs, EOS boundaries
and two-token chunks. This isolates residual replay, not full-backbone equivalence.

MLX and official tokenizers have identical vocabularies and matching token IDs
on all 27,363 tested questions, atomic prompts, subjects and new targets.

Full-decoder logits differ between full and cached execution on the tested key:
maximum differences are 0.25 without edits and 0.875 with edits, although the
edited key's next-token argmax matched. The embedding cache itself was exact.
Framework arithmetic, quantization and transfer of BF16-trained residuals to a
4-bit backbone remain relevant uncertainties; weak transfer is not a general
refutation of EngramEdit. This replay does not isolate their effects.

Text edits supply revised facts, not every reasoning premise or actual retrieval.
Base LongCat retains its native pretrained n-gram memory. Original-answer scores
are retained too: the unedited model should not know fictional updates.

Timing uses fixed condition order and different completion lengths. It does not
establish a speed advantage. Peak Metal allocation is not total system memory.
System swap grew during model loading on this busy 64 GB Mac; model fit alone
does not establish comfortable headroom for longer contexts or edit optimization.

Target optimization, frequency regularization and fitting new ARAX edits are
not implemented in the Mac replay. All completed outcomes are retained.

Receipts: [residual parity](../artifacts/longcat-reference-parity.json),
[tokenizer parity](../artifacts/longcat-tokenizer-parity.json).

`longcat-mquake-30`: **39.30 GB peak Metal allocation**.

`longcat-mquake-smoke-v4`: **39.30 GB peak Metal allocation**.
