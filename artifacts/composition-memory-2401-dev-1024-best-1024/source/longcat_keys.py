"""Portable exact-key matching and MQuAKE scoring, without an MLX dependency.

Matching/scoring follow ModalityDance/EngramEdit at
fb4089be4eec2575847d7e71342ff545ccb70508 (MIT; see third_party/engramedit).
"""

import re


LENGTHS = (2, 3, 4)


def activated_rows(tokens, history, tables, eos_token_id=2):
    """Return one row ID per length/position, including cross-chunk suffixes.

    Row zero is the null update. A dictionary probe, rather than a scan through
    the stored edits, makes matching linear in the number of input tokens.
    """
    if len(tokens) != len(history):
        raise ValueError("Token and history batch sizes differ")
    result = {length: [] for length in LENGTHS}
    for token_row, prefix in zip(tokens, history):
        context = list(prefix[-3:]) + list(token_row)
        offset = len(context) - len(token_row)
        for length in LENGTHS:
            rows = []
            for pos in range(offset, len(context)):
                start = pos - length + 1
                key = tuple(context[start : pos + 1]) if start >= 0 else ()
                valid = len(key) == length and eos_token_id not in key[:-1]
                rows.append(tables[length].get(key, 0) if valid else 0)
            result[length].append(rows)
    return result


def normalize_answer(text):
    return " ".join(text.strip().casefold().split()).strip(" \t\r\n.,;:!?\"'")


def extract_answer(generation, cot=False):
    if cot:
        matches = re.findall(
            r"(?:^|\n)\s*Answer\s*:\s*([^\n]+)", generation, re.IGNORECASE
        )
        return matches[-1].strip() if matches else ""
    first = next((line.strip() for line in generation.splitlines() if line.strip()), "")
    return re.sub(r"^(?:a|answer)\s*:\s*", "", first, flags=re.IGNORECASE)


def matches_answer(prediction, answers):
    value = normalize_answer(prediction)
    return bool(value) and any(value == normalize_answer(answer) for answer in answers)
