import json

import pytest

from sim2gym.protocol import format_observation, parse_reply, render_prompt


@pytest.mark.parametrize("text,expected", [
    ('{"answer":{"array_h":16}}', {"answer": {"array_h": 16}}),
    ('```json\n{"simulate":{"array_h":16}}\n```', {"simulate": {"array_h": 16}}),
    ('Thinking. {"answer":{"nested":{"a":[1,2]}}} trailing text',
     {"answer": {"nested": {"a": [1, 2]}}}),
    ('{"simulate":{}} then {"answer":{"array_h":32}}', {"answer": {"array_h": 32}}),
    ('{"answer":{"array_h":1.6e1}}', {"answer": {"array_h": 16.0}}),
    ('{"reason":"chosen","answer":{"array_h":16}}', {"answer": {"array_h": 16}}),
    ('{"answer":{"text":"a } brace and \\"quote\\""}}',
     {"answer": {"text": 'a } brace and "quote"'}}),
])
def test_parse_valid_replies(text, expected):
    assert parse_reply(text) == expected


@pytest.mark.parametrize("text", [
    "", "no JSON", "null", "42", '"answer"', '{"answer":',
    '{"answer":{},"simulate":{}}', '{"other":{"answer":{}}}',
    '[{"answer":{}}]', '[1,2,3]', '"{\\"answer\\":{}}"',
    '{"answer":{"x":NaN}}', '{"answer":{"x":Infinity}}',
    '{"answer":{"x":-Infinity}}', '{"answer":{"x":1e999}}',
    '{"answer":{"x":[{"y":NaN}]}}', '{"answer":{},"answer":{}}',
    '{"answer":{"x":1,"x":2}}', None,
    '{"answer":{"simulate":{}}', '{broken: {"answer":{}}}',
])
def test_reject_invalid_replies(text):
    assert parse_reply(text) is None


def test_last_valid_top_level_command():
    assert parse_reply('{"answer":{"x":1}} {"metadata":{"answer":{"x":2}}}') == {"answer": {"x": 1}}


def test_observation_json_and_no_nonfinite():
    result = dict(cycles=100, utilization_pct=50.0, stall_cycles=2)
    assert json.loads(format_observation(result)) == result
    with pytest.raises(ValueError):
        format_observation({"cycles": float("nan")})


def test_prompt_is_public_and_describes_protocol():
    task = dict(layer=dict(name="example", kind="gemm", m=8, n=128, k=128, source="source"),
                space=dict(array_h=[8,16], array_w=[8,16], dataflow=["os","ws","is"], sram_split=["balanced"]),
                pe_budget=256, budget=6, optimum={"value": 123456789})
    prompt = render_prompt(task)
    assert "123456789" not in prompt
    assert "256" in prompt and "6 simulator calls" in prompt
    assert '"simulate"' in prompt and '"answer"' in prompt
    assert "cycles" in prompt and "stall_cycles" in prompt and "utilization_pct" in prompt
    assert "JSON object on its last line" in prompt
