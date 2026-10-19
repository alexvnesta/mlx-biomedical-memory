# Pre-test depth-only control

The original registered structural stratum changes depth from 1–2 to 3–4 and
bank size from 10 to 12. Its record IDs also introduce F10/F11, whose learned
codec embeddings receive no training because training worlds use F0 through F9.
That stratum is therefore a joint shift in depth, bank size and two field values.

Before opening any held-out model outcome, we froze a supplementary control at
`artifacts/decisive-v1/depth-only.json`: 96 reasoning worlds with depth 3–4 but
only 10 records and record IDs F0–F9. These reuse the structural stratum's world
seeds and queries, preserve the designed core paths, and reduce decoys. Record
ID assignment and ordering are regenerated independently. All four answer
classes remain balanced. The exact generator asserts each world's answer.

This control changes neither training, development, checkpoint selection nor
the primary registered corpus. It receives the same four selected checkpoints,
with no additional fitting. It separates longer composition from the bank-size
and untrained-record-ID confounds in interpreting this engineered codec.
The original 12-record structural outcomes remain public and separately scored.
The supplementary data and evaluation flag are committed before the first
held-out model run.
