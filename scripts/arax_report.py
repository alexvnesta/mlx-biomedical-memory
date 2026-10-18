"""Render a local, source-linked ARAX evidence browser; no LLM conclusions."""

import argparse
from collections import Counter
import html
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--snapshot", required=True)
p.add_argument("--output", default="runs/arax-evidence.html")
args = p.parse_args()
root = Path(args.snapshot)
summary = json.loads((root / "summary.json").read_text())


def esc(value):
    return html.escape(str(value))


sections = []
for case in summary["cases"]:
    path = root / (
        case["disease_requested"].lower().replace(" ", "-") + ".snapshot.json"
    )
    if not path.exists():
        continue
    d = json.loads(path.read_text())
    nodes = d["nodes"]
    edges = {e["id"]: e for e in d["edges"]}
    cards = []
    for result in d["results"]:
        names = {
            k: ", ".join(nodes.get(i, {}).get("name", i) for i in ids)
            for k, ids in result["node_bindings"].items()
        }
        lines = []
        for relation, ids in result["edge_bindings"].items():
            for eid in ids:
                e = edges[eid]
                sources = ", ".join(
                    s.get("resource_id", "") + " (" + s.get("resource_role", "") + ")"
                    for s in (e.get("sources") or [])
                )
                publications = []
                for a in e.get("attributes") or []:
                    if a.get("attribute_type_id") == "biolink:publications":
                        values = a.get("value", [])
                        publications.extend(
                            values if isinstance(values, list) else [values]
                        )
                links = []
                for value in publications:
                    if str(value).startswith("PMID:"):
                        pmid = str(value).split(":", 1)[1]
                        links.append(
                            f'<a href="https://pubmed.ncbi.nlm.nih.gov/{esc(pmid)}/">{esc(value)}</a>'
                        )
                    else:
                        links.append(esc(value))
                qualifiers = "; ".join(
                    q["qualifier_type_id"].replace("biolink:", "")
                    + "="
                    + q["qualifier_value"]
                    for q in (e.get("qualifiers") or [])
                )
                triple = f"{e.get('subject_name') or e['subject']} → {e['predicate']} → {e.get('object_name') or e['object']}"
                lines.append(
                    f"<li><strong>{esc(triple)}</strong><p>{esc(qualifiers or 'Direction/mechanism qualifiers not supplied')}</p>"
                    f'<p class="meta">{esc(sources)}</p><p>{", ".join(links) or "No publication identifier supplied"}</p></li>'
                )
        title = " / ".join(names.get(k, "") for k in ["chemical", "gene", "disease"])
        cards.append(
            f"<details><summary>{esc(title)}</summary><ul>{''.join(lines)}</ul></details>"
        )
    counts = Counter(w.get("level") for w in d["provider_warnings"])
    diagnostics = " ".join(str(w.get("message", "")) for w in d["provider_warnings"])
    receipt = case["query_receipt"]
    sections.append(
        f"<section><h2>{esc(case['disease_requested'])}</h2>"
        f"<p>{len(d['results'])} returned results · {len(d['edges'])} bound edges · {dict(counts)} provider log messages.</p>"
        '<p class="warning">These are retrieved graph records; primary sources have not been adjudicated. '
        "Broad chemical entities are not necessarily drugs.</p>"
        f'<p class="meta">Provider diagnostics: {esc(diagnostics or "None returned")}</p>'
        f'<p class="meta">Retrieved {esc(receipt["retrieved_utc"])}<br>Response SHA256: {esc(receipt["response_sha256"])}</p>'
        + "".join(cards)
        + "</section>"
    )
document = """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>ARAX rare-disease evidence explorer</title><style>
body{font:16px/1.6 system-ui,sans-serif;max-width:1100px;margin:auto;padding:36px;color:#172c38;background:#f6f8fa}
h1,h2{line-height:1.2}header{border-bottom:3px solid #148b87;padding-bottom:20px}section{margin:34px 0}
details{background:white;border:1px solid #ced9dd;border-radius:8px;padding:12px 16px;margin:10px 0}
summary{cursor:pointer;font-weight:600}li{padding:10px 0;border-bottom:1px solid #eee}p{margin:.5em 0}
.meta{font-size:12px;color:#52656d;overflow-wrap:anywhere}.warning{background:#fff2cf;padding:12px;border-radius:5px}
a{color:#08736f}input{font:inherit;padding:10px;width:95%;border:1px solid #9cbbbd;border-radius:6px}</style>
<header><h1>ARAX rare-disease evidence explorer</h1><p>Fabry disease · Gaucher disease · Rett syndrome</p>
<p>Source-linked graph paths for examination. Connectivity does not establish treatment benefit, efficacy, or safety.
This browser shows the retrieved graph; it is separate from the latent-memory benchmark.</p>
<input id="filter" aria-label="Filter graph results" placeholder="Filter compounds, genes, predicates or sources…"></header>"""
document += "".join(sections)
document += """<script>document.getElementById('filter').addEventListener('input',e=>{const q=e.target.value.toLowerCase();
document.querySelectorAll('details').forEach(d=>d.hidden=!d.textContent.toLowerCase().includes(q));});</script></html>"""
out = Path(args.output)
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(document)
print(out.resolve())
