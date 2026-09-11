from zero_ad_bench.agents.transformers import extract_first_json_object


def test_extracts_plain_json():
    assert extract_first_json_object('{"type":"wait"}') == {"type": "wait"}


def test_extracts_json_after_reasoning_and_unwraps_action():
    text = 'Selected the defensive response.\n```json\n{"action":{"type":"defend","entity_ids":[1],"target_entity_id":10}}\n```'
    assert extract_first_json_object(text) == {"type": "defend", "entity_ids": [1], "target_entity_id": 10}


def test_returns_none_without_object():
    assert extract_first_json_object("wait") is None

