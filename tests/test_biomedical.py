from pathlib import Path
import pytest
from engram_poc.biomedical import build_arax


@pytest.mark.skipif(
    not Path("data/processed/arax-anchored/summary.json").exists(),
    reason="live snapshot kept local; fetch with engram-poc arax",
)
def test_arax_blinding_composition_split_and_connected_negative_controls():
    data = build_arax("data/processed/arax-anchored")
    assert data["audit"]["heldout_disease"] == "Rett syndrome"
    assert not {e.keys for e in data["train"] if e.kind == "compose"} & {
        e.keys for e in data["test"]
    }
    assert all(
        "MONDO:" not in e.prompt() and "CHEBI:" not in e.prompt() for e in data["all"]
    )
    for record in data["provenance"]:
        a, b = record["triples"]
        shared = set((a[0], a[2])) & set((b[0], b[2]))
        assert len(shared) == 1
        for control in record["controls"]:
            assert shared & {control[0], control[2]}
    for split in ["valid", "test"]:
        rows = data[split]
        assert sum(e.label == "recorded" for e in rows) == len(rows) // 2
