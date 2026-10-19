# Trained-memory diagnostic, specified before held-out evaluation

The registered selection rule allows the initial checkpoint to win. At the
512-update development stage, the memory reader's initial checkpoint outscored
the trained candidates. A comparison showing only that fallback would not
answer the user's request to test an actually trained memory interface.

We therefore also evaluate the best **trained** memory candidate, using the same
already-scored development criterion but requiring a positive training step.
The two registered seeds, all primary unseen tests, interventions, binding,
order, masking and the depth-only control are retained. No new fitting or
held-out-based checkpoint choice is allowed. The initial fallback remains
separately reported under the original selection rule.

The trained-only checkpoint and selection receipt are public. The text comparator
is its strongest development-selected checkpoint, including the initial fallback
if it were to outperform all trained readers. Both conditions receive the full
registered training budget and identical sample sequences. Fixed-token latency
compares this trained memory diagnostic with the selected text reader; hashes
identify the exact weights. A failed atomic-readout gate remains a failed
prerequisite, even when the trained diagnostic is evaluated comprehensively.

This diagnostic was added before opening held-out outputs, based only on the
observed development failure. It prevents an untrained selection fallback from
being described as a successfully trained memory system.
