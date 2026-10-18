"""Leakage-audited sign composition over independently addressable atomic edges.

Every atomic edge is available for memory fitting. Only training components have
composed-answer supervision. Validation and test composed answers never enter
the optimizer. The path is supplied; finding a hidden path is a different task.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import random


LABELS = ("positive", "negative")
RULE = (
    "This is a formal signed-graph task, not a claim about treatment. "
    "Each stored edge is positive (+1) or negative (-1). Along a supplied path, "
    "multiply signs. Use this exact table:\n"
    "positive, positive -> positive\nnegative, negative -> positive\n"
    "positive, negative -> negative\nnegative, positive -> negative\n"
)


@dataclass(frozen=True)
class Example:
    id: str
    component: int
    kind: str
    keys: tuple[str, ...]
    signs: tuple[int, ...]
    label: str
    split: str
    task: str = "sign"

    def prompt(self, oracle: bool = False, paraphrase: bool = False) -> str:
        if self.task == "membership":
            facts = ""
            if oracle:
                facts = "Recorded facts:\n" + "".join(
                    f"[{key}] is {'recorded' if value > 0 else 'unobserved'}.\n"
                    for key, value in zip(self.keys, self.signs))
            question = ("Classify this directed atomic edge: " if self.kind == "atomic" else
                        "Are both supplied directed edges recorded in the frozen snapshot? ")
            if paraphrase and self.kind != "atomic":
                question = "Does this supplied pair have two recorded edges in the snapshot? "
            return (
                "This is graph-snapshot membership, not biological truth or treatment efficacy. "
                "An edge is recorded or unobserved. A supplied pair is recorded exactly when BOTH "
                "edges are recorded; otherwise choose unobserved. Unobserved means absent from "
                "this bounded snapshot, not absent from biology.\n" + facts + question +
                " then ".join(f"[{key}]" for key in self.keys) +
                "\nOptions:\nA. recorded\nB. unobserved\nReply with only A or B.")
        facts = ""
        if oracle:
            facts = "Recorded facts:\n" + "".join(
                f"[{key}] has sign {LABELS[sign < 0]}.\n"
                for key, sign in zip(self.keys, self.signs)
            )
        keys = " then ".join(f"[{key}]" for key in self.keys)
        if self.kind == "atomic":
            question = f"What sign is stored for {keys}?"
        elif paraphrase:
            question = f"Compose the signs on this ordered path: {keys}. What is the resulting sign?"
        else:
            question = f"What is the composed sign along the supplied path {keys}?"
        return RULE + facts + question + "\nOptions:\nA. positive\nB. negative\nReply with only A or B."


def build(seed: int = 17, train_components: int = 24, valid_components: int = 8,
          test_components: int = 16) -> dict:
    """Balanced sign combinations within each split; aliases have no meaning."""
    if any(n <= 0 or n % 4 for n in (train_components, valid_components, test_components)):
        raise ValueError("component counts must be positive multiples of four")
    rng = random.Random(seed)
    all_examples = []
    groups = {}
    component = 0
    used_names = set()

    def name(prefix):
        while True:
            token = prefix + "".join(rng.choices("ABCDEFGHJKLMNPQRSTUVWXYZ23456789", k=5))
            if token not in used_names:
                used_names.add(token)
                return token

    for split, count in (("train", train_components), ("valid", valid_components), ("test", test_components)):
        pairs = [(1, 1), (1, -1), (-1, 1), (-1, -1)] * (count // 4)
        rng.shuffle(pairs)
        for left_sign, right_sign in pairs:
            c, g, p = name("Compound_"), name("Gene_"), name("Process_")
            keys = (f"{c} -> {g}", f"{g} -> {p}")
            groups[str(component)] = {"split": split, "entities": [c, g, p], "keys": keys}
            for j, (key, sign) in enumerate(zip(keys, (left_sign, right_sign))):
                all_examples.append(Example(f"atomic-{component}-{j}", component, "atomic", (key,),
                                            (sign,), LABELS[sign < 0], split))
            product = left_sign * right_sign
            all_examples.append(Example(f"compose-{component}", component, "compose", keys,
                                        (left_sign, right_sign), LABELS[product < 0], split))
            component += 1
    # Memorize all atomic premises, including those later needed by held-out paths.
    train = [x for x in all_examples if x.kind == "atomic" or x.split == "train"]
    valid = [x for x in all_examples if x.kind == "compose" and x.split == "valid"]
    test = [x for x in all_examples if x.kind == "compose" and x.split == "test"]
    return {"seed": seed, "labels": LABELS, "groups": groups, "all": all_examples, "train": train,
            "valid": valid, "test": test, "audit": audit(train, valid, test)}


def audit(train, valid, test):
    seen = {tuple(x.keys) for x in train if x.kind == "compose"}
    heldout = {tuple(x.keys) for x in [*valid, *test]}
    atomic = {x.keys[0] for x in train if x.kind == "atomic"}
    if {x.component for x in train if x.kind == "compose"} & {x.component for x in [*valid, *test]}:
        raise ValueError("component leakage")
    if seen & heldout:
        raise ValueError("composed-answer leakage")
    if any(key not in atomic for pair in heldout for key in pair):
        raise ValueError("held-out path lacks an atomic premise")
    return {"heldout_composed_answers_in_training": 0,
            "heldout_atomic_premises_available": True,
            "training_components_disjoint_from_heldout_composition": True,
            "scope": "new compositions of memorized edges; supplied paths; stipulated sign algebra",
            "not_established": ["hidden-path discovery", "biological causality", "drug efficacy"]}


def manifest(dataset):
    rows = [asdict(x) for x in dataset["all"]]
    digest = hashlib.sha256(__import__("json").dumps(rows, sort_keys=True).encode()).hexdigest()
    return {"seed": dataset["seed"], "labels": dataset.get("labels", LABELS), "sha256": digest, "audit": dataset["audit"],
            "counts": {k: len(dataset[k]) for k in ("all", "train", "valid", "test")},
            "examples": rows}
