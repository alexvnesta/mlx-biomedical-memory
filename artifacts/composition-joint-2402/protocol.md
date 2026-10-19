# Joint adaptation with readout retention

This development amendment adds a jointly adapted memory condition before any
final composition generations are opened. The registered frozen-codec/text
comparison and its data remain unchanged and are retained as controls.

The first frozen-codec development candidate retained 63/64 exact atomic
records but produced 0/48 valid development proofs. This is not a final-test
result. Freezing a codec trained only for reconstruction could prevent its
representation from aligning with relationship queries. Test that restriction
rather than treating it as a necessary property of learned memory.

Initialize a third condition from the same repaired readout checkpoint,
SHA-256 `3d913d797cf7209590c88336a014ca44d6a6197fe422f2ecc92d01ef9f3bb3c1`.
Keep the same one-vector codec architecture and query-independent atomic inputs.
Allow both codec and rank-16 LoRA to adapt, with codec learning rate 0.00001
and reader rate 0.00003. Its 5,719,809 updated parameters remain below text's
5,914,624. The base LongCat weights stay frozen. Use the same cases, matched
sample sequences, 64 atomic warmup updates, atomic rehearsal, noisy past atomic
prefixes, two seeds (2401, 2402), and final common update budget selected by the
original development protocol. No new graph test data are used for selection.

Use the same best-NLL/final candidates and actual generated development
selection. Only positive-step candidates with at least 63/64 complete record
readouts are eligible. If this arm fails that gate, do not open its graph test
or interpret reasoning separately from access. Any repair of that development
recipe must be documented. Report all attempts, not only a successful arm.

Freeze all three conditions before opening any final graph generations. Retain
the original frozen-codec arm as an ablation. Regrade identical final worlds,
positive paths, depth strata, interventions, and same-world support readouts.
Report changed codec weights explicitly for this arm; "frozen codec" applies
only to the control, while query independence applies to both memory arms.
The repaired reader's earlier readout test remains a separate, unchanged result.

This amendment tests whether adaptation of the representation matters. It does
not presuppose that joint adaptation succeeds or that latent memory improves
over explicit text, graph traversal, RAG, or clinical evidence tools.
