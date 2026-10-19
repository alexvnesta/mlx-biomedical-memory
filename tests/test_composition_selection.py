from scripts.run_composition_experiment import choose


def test_readout_gate_precedes_reasoning_score_and_excludes_untrained_fallback():
    def candidate(step,atomic,proof,nll):
        return dict(step=step,atomic_correct=atomic,atomic_total=64,reasoning_proofs=proof,weighted_dev_nll=nll)
    inaccessible=candidate(1024,0,48,.001)
    untrained=candidate(0,64,48,.001)
    functional=candidate(512,63,24,.5)
    assert choose([inaccessible,untrained,functional])==functional
    assert choose([inaccessible,untrained]) is None
    better=candidate(1024,64,24,.8)
    assert choose([functional,better])==better
