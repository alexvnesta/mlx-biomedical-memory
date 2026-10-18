import pytest
from engram_poc.arax import normalize, query


def response():
    return {"status": "Success", "message": {
        "knowledge_graph": {"nodes": {"a": {"name": "drug"}, "b": {"name": "gene"},
                                     "c": {"name": "disease"}},
                            "edges": {"e1": {"subject": "a", "object": "b", "predicate": "biolink:affects",
                                             "qualifiers": [{"qualifier_type_id": "biolink:object_direction_qualifier",
                                                             "qualifier_value": "decreased"}],
                                             "sources": [{"resource_id": "infores:chembl", "resource_role": "primary_knowledge_source"}],
                                             "attributes": [{"attribute_type_id": "biolink:publications", "value": ["PMID:123"]}]},
                                      "unbound": {"subject": "b", "object": "c", "predicate": "biolink:treats"}}},
        "results": [{"node_bindings": {"chemical": [{"id": "a"}], "gene": [{"id": "b"}]},
                     "analyses": [{"resource_id": "infores:arax", "edge_bindings": {"chemical_gene": [{"id": "e1"}]}}]}]}}


def test_only_bound_edges_retained_with_direction_qualifiers_and_sources():
    out = normalize(response())
    assert [e["id"] for e in out["edges"]] == ["e1"]
    edge = out["edges"][0]
    assert edge["subject"] == "a" and edge["object"] == "b"
    assert edge["qualifiers"][0]["qualifier_value"] == "decreased"
    assert edge["sources"][0]["resource_id"] == "infores:chembl"
    assert edge["attributes"][0]["value"] == ["PMID:123"]


def test_broken_edge_binding_is_not_silently_accepted():
    data = response()
    data["message"]["results"][0]["analyses"][0]["edge_bindings"]["chemical_gene"] = [{"id": "missing"}]
    with pytest.raises(ValueError, match="missing"):
        normalize(data)


def test_query_preserves_source_relations_without_inventing_treatment():
    q = query("MONDO:0010526", 20)
    assert q["message"]["query_graph"]["nodes"]["disease"]["ids"] == ["MONDO:0010526"]
    assert all("predicates" not in e for e in q["message"]["query_graph"]["edges"].values())
