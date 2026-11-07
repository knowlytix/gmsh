# SPDX-License-Identifier: Apache-2.0
"""Render every assertion node without treating operator anchors as full claims."""

import json


def compact(value):
    if isinstance(value, list):
        return [compact(v) for v in value]
    if isinstance(value, dict):
        if {"document", "start", "end", "text"} <= value.keys():
            return value["text"]
        return {k: compact(v) for k, v in value.items()}
    return value


def render_assertion(assertion):
    nodes = {n["id"]: n for n in assertion["nodes"]}
    if len(nodes) != len(assertion["nodes"]):
        raise ValueError("Duplicate node IDs")
    lines = [f"ASSERTION state={assertion['state']} kind={assertion['kind']}"]
    seen = set()

    def visit(key, depth, ancestors):
        if key in ancestors:
            raise ValueError("Cyclic assertion")
        if key not in nodes:
            raise ValueError("Missing child")
        node = nodes[key]
        seen.add(key)
        fields = {k: compact(v) for k, v in node.items() if k not in ["id", "children"]}
        lines.append("  " * depth + key + " " + json.dumps(fields, ensure_ascii=False))
        for i, child in enumerate(node.get("children", [])):
            lines.append("  " * (depth + 1) + f"operand[{i}] -> {child}")
            visit(child, depth + 2, ancestors | {key})

    visit(assertion["root"], 0, set())
    if seen != set(nodes):
        raise ValueError("Unreachable assertion nodes")
    lines.append(
        "REFERENCES "
        + json.dumps(compact(assertion.get("references", [])), ensure_ascii=False)
    )
    return "\n".join(lines)
