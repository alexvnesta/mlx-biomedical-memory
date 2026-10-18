"""Versioned public ARAX snapshots and bound-edge evidence paths.

The model benchmark and the biomedical case study are separate artifacts.
No association is converted into an inhibitory mechanism or a treatment claim.
"""
from __future__ import annotations
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import ssl
import urllib.error
import urllib.parse
import urllib.request

import truststore


def request(url, directory, stem, body=None, timeout=120):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    raw_body = json.dumps(body, sort_keys=True).encode() if body is not None else None
    req = urllib.request.Request(url, data=raw_body, headers={"Accept": "application/json",
                                                            "Content-Type": "application/json"})
    receipt = {"url": url, "retrieved_utc": datetime.now(timezone.utc).isoformat(),
               "request": body}
    context = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=context) as response:
            raw = response.read()
            receipt["http_status"] = response.status
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        receipt["http_status"] = exc.code
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        receipt.update(ok=False, error=str(exc))
        (directory / f"{stem}.receipt.json").write_text(json.dumps(receipt, indent=2))
        return None, receipt
    receipt["response_sha256"] = hashlib.sha256(raw).hexdigest()
    (directory / f"{stem}.response.json").write_bytes(raw)
    try:
        data = json.loads(raw)
    except (ValueError, UnicodeDecodeError):
        receipt.update(ok=False, error="invalid JSON")
        data = None
    else:
        receipt["ok"] = receipt["http_status"] == 200 and isinstance(data, dict)
        if isinstance(data, dict):
            receipt["provider_status"] = data.get("status")
            receipt["description"] = data.get("description")
    (directory / f"{stem}.receipt.json").write_text(json.dumps(receipt, indent=2))
    return data, receipt


def normalize(response):
    """Keep query-bound edges and their supplied provenance, not every KG edge."""
    message = response.get("message") or {}
    kg = message.get("knowledge_graph") or {}
    nodes, edges = kg.get("nodes") or {}, kg.get("edges") or {}
    results = message.get("results") or []
    retained = {}
    paths = []
    for number, result in enumerate(results):
        bound_nodes = {k: [r["id"] for r in v] for k, v in (result.get("node_bindings") or {}).items()}
        for analysis in result.get("analyses") or []:
            bound = {k: [r["id"] for r in v] for k, v in (analysis.get("edge_bindings") or {}).items()}
            for ids in bound.values():
                for edge_id in ids:
                    edge = edges.get(edge_id)
                    if not edge or edge.get("subject") not in nodes or edge.get("object") not in nodes:
                        raise ValueError("ARAX result binds a missing edge or node")
                    retained[edge_id] = {"id": edge_id, **edge,
                                         "subject_name": nodes[edge["subject"]].get("name"),
                                         "object_name": nodes[edge["object"]].get("name")}
            paths.append({"result": number, "node_bindings": bound_nodes,
                          "edge_bindings": bound, "resource_id": analysis.get("resource_id"),
                          "score": analysis.get("score"), "support_graphs": analysis.get("support_graphs", [])})
    logs = [row for row in response.get("logs", [])
            if row.get("level") in {"ERROR", "WARNING"}]
    return {"status": response.get("status"), "description": response.get("description"),
            "schema_version": response.get("schema_version"),
            "biolink_version": response.get("biolink_version"),
            "nodes": {key: nodes[key] for key in {n for e in retained.values() for n in (e["subject"], e["object"]) }},
            "edges": list(retained.values()), "results": paths, "provider_warnings": logs,
            "auxiliary_graphs": message.get("auxiliary_graphs") or {},
            "scope": "retrieved query-bound graph paths; primary claims not independently verified",
            "absence_means": "not observed in this bounded response, not biological absence"}


def query(disease_id, max_results):
    # Narrow monogenic case studies to the canonical gene observed in the initial
    # ARAX response. This avoids broad family concepts such as 'Enzymes'.
    anchors = {"MONDO:0010526": "NCBIGene:2717", "MONDO:0018150": "NCBIGene:2629",
               "MONDO:0010726": "NCBIGene:4204"}
    gene = {"categories": ["biolink:Gene"]}
    if disease_id in anchors:
        gene["ids"] = [anchors[disease_id]]
    return {"message": {"query_graph": {
        "nodes": {"disease": {"ids": [disease_id]}, "gene": gene,
                  "chemical": {"categories": ["biolink:ChemicalEntity"]}},
        "edges": {"gene_disease": {"subject": "gene", "object": "disease"},
                  "chemical_gene": {"subject": "chemical", "object": "gene"}}}},
        "operations": {"actions": ["expand()", "resultify()",
            f"filter_results(action=limit_number_of_results,max_results={max_results})",
            "return(message=true,store=false)"]}}


def fetch_cases(args):
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    raw_dir = root / "raw"
    summary = {"endpoint": args.endpoint, "cases": [],
               "scope": "rare-disease graph evidence exploration, not therapeutic recommendations"}
    for name in args.diseases:
        stem = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
        entity, er = request(args.endpoint + "/entity?" + urllib.parse.urlencode({"q": name}),
                             raw_dir, stem + "-entity", timeout=args.timeout)
        identity = ((entity or {}).get(name) or {}).get("id") or {}
        case = {"disease_requested": name, "identity": identity, "entity_receipt": er}
        if not identity.get("identifier"):
            case.update(status="entity_resolution_failed")
        else:
            response, receipt = request(args.endpoint + "/query", raw_dir, stem + "-query",
                                        query(identity["identifier"], args.max_results), args.timeout)
            case["query_receipt"] = receipt
            if response is not None and receipt["ok"]:
                snapshot = normalize(response)
                (root / f"{stem}.snapshot.json").write_text(json.dumps(snapshot, indent=2))
                case.update(status=snapshot["status"], edges=len(snapshot["edges"]),
                            results=len(snapshot["results"]),
                            predicates=dict(Counter(edge["predicate"] for edge in snapshot["edges"])),
                            description=snapshot["description"],
                            snapshot=str(root / f"{stem}.snapshot.json"))
            else:
                case.update(status="request_failed")
        summary["cases"].append(case)
        (root / "summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps({"disease": name, "status": case["status"], "edges": case.get("edges", 0)}), flush=True)
    return summary
