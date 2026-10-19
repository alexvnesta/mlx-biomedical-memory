# Separating path search from combining stored values

This diagnostic uses the final fixed readers, without selecting or updating any
weights. It is separate from the primary free TRACE comparison. A graph oracle
supplies a valid directed path's record IDs; it does **not** supply the path's
sign or the final answer. The model must predict the next UP/DOWN token at the
path's `sign` field. This is a conditional decision, not a free generated proof.

Generate eight fresh worlds at each path depth one through four, using ten
records, new world namespaces, and final-query endpoint partition 1. Only
single-path worlds are used, so ordering of opposite-sign paths cannot reveal
the answer. Hold topology, qualifiers, IDs, question, and supplied path fixed.
Enumerate all 2^depth sign assignments on that path. For each assignment also
flip one predetermined irrelevant edge's sign; the expected product must stay
the same. A reader that merely copies one edge or multiplies every bank edge
cannot pass all these settings. Labels are balanced within each world.

The supplied assistant prefix ends at the opening quote of `sign`, before any
output sign or answer token. Cache only frozen states of this **input** prefix.
The oracle path IDs are part of the diagnostic input, not an answer withheld by
this task. No sign-target state is computed or cached. Memory-prefix token IDs
must be identical across every sign setting in a world. Only codec input values
change. Text receives the same changed facts explicitly.

Report sign-token accuracy and the number of worlds with **every** path-sign
setting and irrelevant-edge flip correct. Report all depths and both fitted
optimization seeds. Masking the memory is an additional control. Do not describe
success here as discovering a path, succeeding at the primary graph question,
or establishing an advantage over a graph that can execute the same algebra.
It tests whether composition is available when retrieval is supplied, helping
distinguish content search from using the retrieved values.

The task is stipulated algebra on complete synthetic worlds. It is not a claim
that biological effects compose by sign multiplication or that a graph path
establishes causality. Results do not select checkpoints or tune the final test.
