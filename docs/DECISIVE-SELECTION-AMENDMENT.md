# Retain earlier generated-development candidates

Before opening any held-out outcome, checkpoint selection was clarified to retain
all candidates already scored on development data during the 512-update stage
when selecting after the 1,024-update extension. Otherwise the lower-NLL
checkpoint at the larger budget can discard an earlier reader with better
free-running proofs. That would give a previously observed better text or memory
checkpoint an unfair disadvantage simply because training continued.

Selection still maximizes development reasoning proofs, then exact atomic
readout, then lower development NLL. It uses no additional data or held-out score.
The earlier receipts, candidates and raw development generations are preserved.
The common 1,024-update cap and stopping rule remain unchanged. The amended driver
unions the earlier scored candidates with the initial, best-loss and final-budget
candidates. It applies identically to text and memory.
