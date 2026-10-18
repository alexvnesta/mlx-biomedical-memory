# Position relative to existing work

This repository does not claim to introduce conditional memory, graph-to-LLM
interfaces, knowledge editing, or biomedical graph reasoning. It implements a
small, auditable combination on Apple silicon and tests a narrower question.

| Work | Existing contribution relevant here | Difference in this prototype |
|---|---|---|
| [DeepSeek Engram](https://arxiv.org/abs/2601.07372) | Conditional hashed n-gram memory and context-dependent fusion | Explicit atomic edge phrases, one table with open addressing, scalar gate, frozen Qwen3-4B |
| [TinyEngram](https://github.com/AutoArk/TinyEngram) | Added conditional modules and biomedical adaptation of a small frozen model | MLX implementation and held-out composition controls rather than general biomedical text adaptation |
| [EngramEdit](https://arxiv.org/abs/2610.10533) | Decoupled factual editing and tests of paraphrase/multi-hop use | This prototype fits added memory; it does not reproduce the editing algorithm |
| [G-Retriever](https://arxiv.org/abs/2402.07630) | Learned graph-to-LLM interface with retrieval and soft prompting | No GNN or hidden graph traversal; supplied paths activate independently learned atomic rows |
| [ARAX](https://pmc.ncbi.nlm.nih.gov/articles/PMC10027432/) | Standards-based biomedical graph querying and reasoning | Used as an evidence source and snapshot benchmark, with original predicate/qualifier metadata retained separately |
| [XMemTransfer](https://arxiv.org/abs/2608.17050) | Frozen external-memory transfer through adapted target readers | Only one backbone is studied here; memory portability is not evaluated |

An architectural result here would require reliable held-out composition beyond
disabled/shuffled memory, followed by an accuracy or fairly measured efficiency
advantage over an identical model supplied with text facts.
A biomedical result would require substantially more: context-sensitive relation
semantics, reliable evidence attribution, broad disease holdouts, and external
adjudication. A positive Boolean snapshot-membership result alone would not meet
that standard.

The most useful early outcomes can be negative: identifying memory encodings that
memorize training examples without transferring the operation, or showing that
traditional LLM+KG already provides a better solution at this scale. All completed
runs are retained so those outcomes remain visible.
