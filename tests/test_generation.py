from generation import _parse_citations, _parse_generation_json


def test_parse_generation_json_strict():
    obj = _parse_generation_json('{"answer": "hi", "citations": [1]}')
    assert obj["answer"] == "hi"
    assert obj["citations"] == [1]


def test_parse_generation_json_extracts_object():
    obj = _parse_generation_json('Here is JSON: {"answer": "ok", "citations": []} end')
    assert obj["answer"] == "ok"


def test_parse_citations_dedupes():
    assert _parse_citations([1, 1, 2, "3", 0, -1]) == [1, 2, 3]
