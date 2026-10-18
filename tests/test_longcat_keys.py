from engram_poc.longcat_keys import activated_rows, extract_answer, matches_answer


def test_chunked_matching_equals_full_sequence_and_stops_at_eos():
    tables = {2: {(4, 5): 1, (2, 4): 2}, 3: {(3, 4, 5): 3}, 4: {(2, 3, 4, 5): 4}}
    full = activated_rows([[2, 3, 4, 5, 2, 4, 5]], [[]], tables)
    first = activated_rows([[2, 3, 4]], [[]], tables)
    tail = activated_rows([[5, 2, 4, 5]], [[2, 3, 4]], tables)
    for length in tables:
        assert first[length][0] + tail[length][0] == full[length][0]
    assert full[2][0] == [0, 0, 0, 1, 0, 0, 1]
    assert full[3][0][3] == 3
    assert full[4][0][3] == 0  # EOS inside a key blocks the update.


def test_official_answer_extraction_uses_last_cot_answer_and_exact_aliases():
    assert extract_answer(" Thoughts here\nAnswer: Paris\n", cot=True) == "Paris"
    assert extract_answer("Answer: London\nAnswer: Paris", cot=True) == "Paris"
    assert extract_answer("No explicit answer", cot=True) == ""
    assert matches_answer(extract_answer("A: PARIS.\nextra"), ["Paris"])
    assert not matches_answer("Paris is plausible", ["Paris"])
