# Handling a failed retention gate

This amendment is recorded before any final graph generations. The second
frozen-codec fit completed development with 62/64 exact atomic records for
both registered candidates. It fails the unchanged 63/64 eligibility rule.
Its test remains closed: a reasonably high aggregate readout rate does not
justify moving a declared gate after observing a failure.

The first frozen-codec seed is eligible. The joint-adaptation arm and the text
control can proceed independently, using the already chosen common budget of
2048 updates. The joint arm is still initialized from the repaired checkpoint
and must independently pass 63/64 for each seed. No untrained fallback, changed
threshold, replacement seed or tuning from final outputs is permitted.

After all development fits finish, publish every outcome, including failures.
Freeze the eligible checkpoints before any final graph outputs. Evaluate only
eligible reader/seed pairs. Report missing pairs as failed retention, never as
zero reasoning accuracy, and give every denominator and per-seed result.
Pairwise comparisons and support-readable subsets use only common eligible
seeds. If no memory fit qualifies, keep memory reasoning tests closed and
report that qualification failure rather than drawing a reasoning conclusion.

The original controller, registration and failed gate are preserved unchanged.
A failed all-arms gate cannot erase an independently qualified readout result.
This amendment changes orchestration and reporting, not training data, model
selection, supervision, update budgets or final cases.
