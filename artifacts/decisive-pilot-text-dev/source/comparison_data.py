"""Fixed controlled tasks for quantized LongCat; no composed answers are fitted."""

import hashlib
import json
from pathlib import Path
import random
import re


def expressions(prompt, subject, question=None):
    variants = [prompt, "According to the record, " + prompt,
                "In this reference, " + prompt, "The stored fact states that " + prompt]
    if question and question.count(subject) == 1:
        variants.append(question.replace(subject, "{}") + " Answer:")
    return list(dict.fromkeys(variants))


def build(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    facts, questions, provenance = [], [], {}
    mquake_path = Path("data/processed/longcat/MQuAKE-CF-3k-v2.json")
    mquake = json.loads(mquake_path.read_text())
    ids = list(range(len(mquake)))
    random.Random(73).shuffle(ids)
    selected = []
    for hops in (2, 3, 4):
        selected.extend([i for i in ids if len(mquake[i]["new_single_hops"]) == hops][:2])
    for case_id in selected:
        case = mquake[case_id]
        fact_ids = []
        for i, edit in enumerate(case["requested_rewrite"]):
            identifier = f"mquake-{case_id}-{i}"
            fact = dict(id=identifier, suite="mquake", component=str(case_id),
                subject=edit["subject"], prompt=edit["prompt"], target=edit["target_new"]["str"],
                expressions=expressions(edit["prompt"], edit["subject"], edit.get("question")))
            facts.append(fact)
            fact_ids.append(identifier)
            for variant, text in enumerate([edit.get("question", edit["prompt"].format(edit["subject"])),
                    "What does the reference give as the missing value: " + edit["prompt"].format(edit["subject"]) + "?"]):
                questions.append(dict(id=f"{identifier}-atomic-{variant}", suite="mquake", component=str(case_id),
                    kind="recall" if variant == 0 else "paraphrase", text=text,
                    answers=[fact["target"]], needed=[identifier], entity=edit["subject"]))
        # Include unchanged premises too, so every method has the complete chain.
        # Their labels are not composed-answer supervision.
        chain_ids = []
        root_subject = None
        for hop_index, hop in enumerate(case['new_single_hops']):
            cloze = hop['cloze']
            if hop_index:
                subject = case['new_single_hops'][hop_index-1]['answer']
            elif cloze.startswith('The company that produced '):
                subject = cloze.removeprefix('The company that produced ').removesuffix(' is')
            else:
                subject = re.split(r' (?:is|was|plays) ', cloze, maxsplit=1)[0]
            if subject not in cloze:
                raise ValueError('Could not identify the explicit chain subject')
            if root_subject is None:
                root_subject = subject
            existing = next((f for f in facts if f['component']==str(case_id) and f['subject']==subject and f['target']==hop['answer']), None)
            if existing is None:
                identifier = f'mquake-{case_id}-support-{hop_index}'
                prompt = cloze.replace(subject, '{}', 1)
                existing = dict(id=identifier, suite='mquake', component=str(case_id),
                    subject=subject, prompt=prompt, target=hop['answer'],
                    expressions=expressions(prompt, subject, hop['question']), unchanged_support=True)
                facts.append(existing)
                questions.append(dict(id=identifier+'-atomic', suite='mquake', component=str(case_id),
                    kind='recall', text=hop['question'], answers=[hop['answer'], *hop.get('answer_alias',[])],
                    needed=[identifier], entity=subject))
            chain_ids.append(existing['id'])
        for i, text in enumerate(case["questions"]):
            questions.append(dict(id=f"mquake-{case_id}-compose-{i}", suite="mquake", component=str(case_id),
                kind="composition", text=text, answers=[case["new_answer"], *case.get("new_answer_alias", [])],
                needed=chain_ids, entity=root_subject, hops=len(case["new_single_hops"])))
    # A deliberately functional selected-neighbor VIEW of real snapshot edges.
    # It is not a claim that chemicals have only one gene target or imply treatment.
    cases = ["fabry-disease", "gaucher-disease", "rett-syndrome"]
    genes = ["Gene Neral Vosk", "Gene Temar Solvik", "Gene Pavor Kelmic"]
    diseases = ["Disease Vela", "Disease Cedar", "Disease Quartz"]
    chemicals = ["Chemical Mira Tovak", "Chemical Lumar Kevon", "Chemical Rivo Malken",
                 "Chemical Sova Trevik", "Chemical Pira Nolvak", "Chemical Daro Vesnik"]
    used_chemicals = set()
    for case_index, slug in enumerate(cases):
        path = Path("data/processed/arax-anchored") / f"{slug}.snapshot.json"
        snapshot = json.loads(path.read_text())
        candidates = [e for e in snapshot["edges"] if e["object"].startswith("NCBIGene:")
                      and e["subject"].startswith(("CHEBI:", "PUBCHEM.COMPOUND:", "CHEMBL.COMPOUND:"))]
        selected_edges = []
        for edge in candidates:
            if edge["subject"] not in used_chemicals:
                selected_edges.append(edge)
                used_chemicals.add(edge["subject"])
            if len(selected_edges) == 2:
                break
        if len(selected_edges) != 2:
            raise ValueError("Insufficient distinct chemical anchors")
        gene_id = selected_edges[0]["object"]
        if any(e["object"] != gene_id for e in selected_edges):
            raise ValueError("Selected chemical records must share the anchored gene")
        disease_edges = [e for e in snapshot["edges"] if
            (e["subject"] == gene_id and e["object"].startswith("MONDO:")) or
            (e["object"] == gene_id and e["subject"].startswith("MONDO:"))]
        if not disease_edges:
            raise ValueError("Missing recorded gene/disease connection")
        gene_alias, disease_alias = genes[case_index], diseases[case_index]
        gene_fact_id = f"arax-{slug}-gene"
        prompt = "In the snapshot view, the selected disease neighbor of {} is"
        question = "Which selected disease neighbor is recorded for " + gene_alias + "?"
        gene_fact = dict(id=gene_fact_id, suite="arax", component=slug, subject=gene_alias,
            target=disease_alias, relation="selected_disease", prompt=prompt,
            expressions=expressions(prompt, gene_alias, question))
        facts.append(gene_fact)
        provenance[gene_fact_id] = dict(snapshot=str(path), snapshot_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            source_edge=disease_edges[0], alias_subject=gene_alias, alias_object=disease_alias)
        questions.append(dict(id=gene_fact_id+"-recall", suite="arax", component=slug,
            kind="recall", text=question, answers=[disease_alias], needed=[gene_fact_id], entity=gene_alias))
        for offset, edge in enumerate(selected_edges):
            chemical = chemicals[2*case_index+offset]
            identifier = f"arax-{slug}-chemical-{offset}"
            prompt = "In the snapshot view, the selected gene neighbor of {} is"
            question = "Which selected gene neighbor is recorded for " + chemical + "?"
            fact = dict(id=identifier, suite="arax", component=slug, subject=chemical,
                target=gene_alias, relation="selected_gene", prompt=prompt,
                expressions=expressions(prompt, chemical, question))
            facts.append(fact)
            provenance[identifier] = dict(snapshot=str(path), snapshot_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                source_edge=edge, alias_subject=chemical, alias_object=gene_alias)
            questions.append(dict(id=identifier+"-recall", suite="arax", component=slug,
                kind="recall", text=question, answers=[gene_alias], needed=[identifier], entity=chemical))
            questions.append(dict(id=identifier+"-paraphrase", suite="arax", component=slug,
                kind="paraphrase", text="Retrieve the gene label selected as a neighbor for " + chemical + ".",
                answers=[gene_alias], needed=[identifier], entity=chemical))
            for variant, text in enumerate([
                f"Starting from {chemical}, follow selected_gene then selected_disease. What disease label do you reach?",
                f"What is the selected disease neighbor of the selected gene neighbor of {chemical}?",
                f"Use two recorded links from {chemical}: first its selected gene, then that gene's selected disease. Return the disease label.",
            ]):
                questions.append(dict(id=f"{identifier}-compose-{variant}", suite="arax", component=slug,
                    kind="composition", text=text, answers=[disease_alias], needed=[identifier, gene_fact_id], entity=chemical, hops=2))
    for i in range(6):
        entity = f"Chemical Unknown Vezik {i}"
        questions.append(dict(id=f"unknown-{i}", suite="arax", component=f"unknown-{i}", kind="unknown",
            text=f"What selected gene neighbor does the frozen snapshot view record for {entity}?",
            entity=entity, needed=[], answers=["UNKNOWN"]))
    # Unrelated generic questions test behavioral retention rather than untouched weights alone.
    locality = [
        ("What is the capital of Spain?", ["Madrid"]), ("What is 7 plus 8?", ["15", "fifteen"]),
        ("Which planet is known as the Red Planet?", ["Mars"]),
        ("What is the chemical symbol for gold?", ["Au"]),
        ("What is the largest ocean on Earth?", ["Pacific", "Pacific Ocean", "the Pacific Ocean"]),
        ("How many sides does a triangle have?", ["3", "three"]),
        ("What gas do plants absorb during photosynthesis?", ["carbon dioxide", "CO2"]),
        ("What is the square root of 81?", ["9", "nine"]),
    ]
    for i, (text, answers) in enumerate(locality):
        questions.append(dict(id=f"locality-{i}", suite="locality", component=str(i),
            kind="locality", text=text, answers=answers, entity="", needed=[]))
    for fact in facts:
        fact["text"] = fact["prompt"].format(fact["subject"]) + " " + fact["target"] + "."
    # Distractors share biomedical retrieval vocabulary, but have disjoint entities.
    distractors = [dict(id=f"distractor-{i}", subject=f"Chemical Decoy {i}", target=f"Gene Decoy {i}",
        text=f"In a different snapshot view, the selected gene neighbor of Chemical Decoy {i} is Gene Decoy {i}.")
        for i in range(120)]
    corpus = dict(facts=facts, distractors=distractors, questions=questions,
        selection=dict(seed=73, mquake_case_ids=selected, mquake_hop_counts=[2, 2, 3, 3, 4, 4]),
        protocol=dict(steps=25, lr=0.5, contexts=["{}", "This reference states: {}"],
            expressions="canonical plus three fixed prefix variants plus supplied atomic question",
            fitting="atomic targets only; no composed targets or test queries in optimization",
            arax_scope="anonymized functional selected-neighbor projection of retained records; not biological causality",
            mquake_scope="fictional counterfactual facts, not current-world truth",
            precision="4-bit frozen native LongCat; newly fitted BF16 additive residuals"))
    (output/"dataset.json").write_text(json.dumps(corpus, indent=2)+"\n")
    (output/"arax-provenance-local.json").write_text(json.dumps(provenance, indent=2)+"\n")
    print(json.dumps(dict(facts=len(facts), questions=len(questions), mquake_cases=selected)))
    return corpus


if __name__ == "__main__":
    build("data/processed/comparison-v1")
