from dataclasses import replace
import pytest
from engram_poc.benchmark import build, audit, manifest


def test_heldout_answers_absent_but_atomic_premises_available():
    data = build()
    assert not ({x.component for x in data["train"] if x.kind == "compose"} &
                {x.component for x in data["test"]})
    atoms = {x.keys[0] for x in data["train"] if x.kind == "atomic"}
    assert all(k in atoms for x in data["test"] for k in x.keys)
    assert sum(x.label == "positive" for x in data["test"]) == len(data["test"]) // 2
    assert manifest(data)["sha256"] == manifest(build())["sha256"]


def test_composition_leakage_fails_closed():
    data = build()
    with pytest.raises(ValueError, match="leakage"):
        audit(data["train"] + [data["test"][0]], data["valid"], data["test"])


def test_labels_follow_rule_and_normal_prompt_hides_premises():
    data = build()
    for ex in data["test"]:
        assert ex.label == ("positive" if ex.signs[0] * ex.signs[1] > 0 else "negative")
        assert "Recorded facts:" not in ex.prompt()
        assert "Recorded facts:" in ex.prompt(oracle=True)


def test_counterfactual_edge_changes_only_affected_expected_answer():
    data = build()
    ex = data["test"][0]
    changed = replace(ex, signs=(-ex.signs[0], ex.signs[1]),
                      label="negative" if ex.label == "positive" else "positive")
    assert changed.keys == ex.keys
    assert changed.label != ex.label
    assert data["test"][1:] == build()["test"][1:]
