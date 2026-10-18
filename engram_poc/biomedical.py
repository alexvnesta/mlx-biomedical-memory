"""Build blinded, held-out composition cases from actual ARAX bound edges.

Negative examples are unobserved in a bounded snapshot, not negative biological
evidence. Only individual edge memberships are written into conditional memory.
The supplied two-edge path is never found by hidden symbolic traversal.
"""
from __future__ import annotations
from dataclasses import asdict
import json
from pathlib import Path
import random

from .benchmark import Example, audit


def edge_tuple(edge):
    return edge["subject"], edge["predicate"], edge["object"]


def build_arax(directory, seed=17, max_paths_per_case=12):
    files = [("fabry-disease", "train"), ("gaucher-disease", "valid"), ("rett-syndrome", "test")]
    snapshots = {stem: json.loads((Path(directory) / f"{stem}.snapshot.json").read_text())
                 for stem, _ in files}
    if any(d.get("status") not in ("Success", "OK") for d in snapshots.values()):
        raise ValueError("ARAX snapshots must report success")
    nodes = {key: value for snap in snapshots.values() for key, value in snap["nodes"].items()}
    triples = {edge_tuple(edge) for snap in snapshots.values() for edge in snap["edges"]}
    rng = random.Random(seed)
    names = sorted(nodes)
    shuffled_names = names[:]
    rng.shuffle(shuffled_names)
    aliases = {curie: f"Entity_{i:05d}" for i, curie in enumerate(shuffled_names)}
    def key(triple):
        a, p, b = triple
        return f"{aliases[a]} | {p} | {aliases[b]}"
    def category(curie):
        # CURIE families are a conservative corruption pool; both endpoints remain real IDs.
        return curie.split(":")[0]
    def corrupt(triple, bridge):
        a, p, b = triple
        candidates = [(c, p, b) for c in names if a != bridge and category(c) == category(a) and c != a]
        candidates += [(a, p, c) for c in names if b != bridge and category(c) == category(b) and c != b]
        rng.shuffle(candidates)
        return next((t for t in candidates if t not in triples), None)
    atoms = {}
    composed = []
    provenance = []
    component = 0
    seen_paths = set()
    for stem, split in files:
        snap = snapshots[stem]
        edges = {edge["id"]: edge for edge in snap["edges"]}
        accepted = 0
        for result in snap["results"]:
            bound = result["edge_bindings"]
            if not bound.get("chemical_gene") or not bound.get("gene_disease"):
                continue
            left = edges[bound["chemical_gene"][0]]
            right = edges[bound["gene_disease"][0]]
            # Exclude broad family concepts masquerading as the query's gene slot.
            genes = result["node_bindings"].get("gene", [])
            if len(genes) != 1 or not genes[0].startswith("NCBIGene:"):
                continue
            if genes[0] not in {left["subject"], left["object"]} & {right["subject"], right["object"]}:
                continue
            a, b = edge_tuple(left), edge_tuple(right)
            if (a, b) in seen_paths:
                continue
            # Preserve the shared gene in negative controls, so disconnected
            # endpoint text cannot reveal the answer without accessing memory.
            ca, cb = corrupt(a, genes[0]), corrupt(b, genes[0])
            if ca is None or cb is None:
                continue
            seen_paths.add((a, b))
            for triple, state in ((a, 1), (b, 1), (ca, -1), (cb, -1)):
                k = key(triple)
                ex = Example(f"atomic-arax-{len(atoms)}", component, "atomic", (k,), (state,),
                             "recorded" if state > 0 else "unobserved", split, "membership")
                if k in atoms and atoms[k].label != ex.label:
                    raise ValueError("contradictory membership labels")
                atoms.setdefault(k, ex)
            # One positive and one of each negative sign combination per path, with
            # equal total class weight by using three repeats of the positive.
            combinations = [(a,b,1,1)] * 3 + [(a,cb,1,-1), (ca,b,-1,1), (ca,cb,-1,-1)]
            for j, (x,y,sx,sy) in enumerate(combinations):
                composed.append(Example(f"compose-arax-{component}-{j}", component, "compose",
                    (key(x),key(y)), (sx,sy), "recorded" if sx == sy == 1 else "unobserved", split, "membership"))
            provenance.append({"component": component, "case": stem, "split": split,
                               "edge_ids": [left["id"], right["id"]], "triples": [a,b],
                               "controls": [ca,cb], "response_sources": [left.get("sources"),right.get("sources")]})
            component += 1
            accepted += 1
            if accepted >= max_paths_per_case:
                break
    all_examples = list(atoms.values()) + composed
    train = list(atoms.values()) + [e for e in composed if e.split == "train"]
    valid = [e for e in composed if e.split == "valid"]
    test = [e for e in composed if e.split == "test"]
    if not valid or not test or not any(e.kind == "compose" for e in train):
        raise ValueError("insufficient qualified paths in at least one ARAX case")
    check = audit(train, valid, test)
    check.update(scope="new combinations of supplied directed edge memberships in a bounded ARAX snapshot",
                 heldout_disease="Rett syndrome", blinded_entity_identifiers=True,
                 negative_labels="not observed in union of retained bound edges; not biological negatives",
                 not_established=["hidden-path discovery", "relation causality", "drug efficacy", "novel repurposing"])
    return {"seed":seed, "labels":("recorded","unobserved"), "all":all_examples,
            "train":train, "valid":valid, "test":test, "audit":check,
            "aliases":aliases, "provenance":provenance}
