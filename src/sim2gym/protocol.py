"""Plain JSON interaction protocol for budgeted accelerator design."""

import json
import math

from sim2gym.packs import accel

RETRY_MESSAGE = ('Invalid reply. End your reply with one JSON object on the last line: '
                 '{"simulate": {"array_h": ..., "array_w": ..., "dataflow": ..., "sram_split": ...}} '
                 'or {"answer": {"array_h": ..., "array_w": ..., "dataflow": ..., "sram_split": ...}}.')


def render_prompt(task: dict) -> str:
    choices = "\n".join(f"- {key}: {json.dumps(values)}" for key, values in task["space"].items())
    return (f"Design an accelerator for {accel.describe_layer(task['layer'])}\n"
            f"Choose exactly these fields with the allowed values:\n{choices}\n"
            f"PE budget: {task['pe_budget']}; array height times width must fit this budget.\n"
            "Objective: minimize simulator cycles. Total SRAM is fixed at 192 kB; "
            "balanced assigns 64 kB each to input, filter and output. Memory bandwidth is fixed.\n"
            f"You may make at most {task['budget']} simulator calls. "
            "Each simulate request uses one call and returns cycles, utilization_pct, and stall_cycles.\n"
            'Use {"simulate": {"array_h": ..., "array_w": ..., "dataflow": ..., "sram_split": ...}} '
            'to experiment. Use {"answer": {"array_h": ..., "array_w": ..., "dataflow": ..., "sram_split": ...}} '
            "to submit your final design and end the episode.\n"
            "End every reply with exactly one JSON object on its last line.")


def _finite(value):
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite(item) for item in value)
    return True


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def _skip_container(text, start):
    depth = 0
    quoted = escaped = False
    for pos in range(start, len(text)):
        char = text[pos]
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
        elif char == '"':
            quoted = True
        elif char in "{[":
            depth += 1
        elif char in "}]":
            depth -= 1
            if depth == 0:
                return pos + 1
    return len(text)


def parse_reply(text: str) -> dict | None:
    if not isinstance(text, str):
        return None
    decoder = json.JSONDecoder()
    strict = json.JSONDecoder(object_pairs_hook=_unique_object)
    last = None
    pos = 0
    while pos < len(text):
        if text[pos] not in '{["':
            pos += 1
            continue
        try:
            value, end = decoder.raw_decode(text, pos)
        except (ValueError, RecursionError):
            pos = _skip_container(text, pos) if text[pos] in "{[" else pos + 1
            continue
        # Advance past whole containers so nested commands cannot escape their envelope.
        if isinstance(value, dict) and len(set(value) & {"simulate", "answer"}) == 1:
            try:
                strict.raw_decode(text, pos)
                if _finite(value):
                    key = "simulate" if "simulate" in value else "answer"
                    last = {key: value[key]}
            except (ValueError, RecursionError):
                pass
        pos = end
    return last


def format_observation(result: dict) -> str:
    return json.dumps(result, sort_keys=True, allow_nan=False)
