"""Provider-compatible strict schema for one adaptive research action."""

from __future__ import annotations

from copy import deepcopy

from pydantic import BaseModel

WORKER_WIRE_VERSION = "single_action_grounded_v3"


def worker_response_schema(
    schema: type[BaseModel],
    read_spans: list[dict] | None = None,
    *,
    final_action: bool = False,
    strict_tools: bool = False,
) -> dict:
    """Close dynamic argument keys on the wire; preserve domain validation locally.

    Strict provider schemas require all object fields to be declared and required.
    Unused action arguments are nullable and removed by WorkerAction validation.
    Facts, source spans, and card validation are otherwise unchanged.
    """
    body = deepcopy(schema.model_json_schema())
    types = {
        "query": "string",
        "symbol": "string",
        "scope": "string",
        "evidence_id": "string",
        "start": "integer",
        "end": "integer",
        "top_n": "integer",
    }
    body["properties"]["arguments"] = {
        "type": "object",
        "properties": {key: {"type": [kind, "null"]} for key, kind in types.items()},
        "additionalProperties": False,
        "required": list(types),
    }

    def close(node):
        if isinstance(node, dict):
            node.pop("default", None)
            if node.get("type") == "object":
                node["additionalProperties"] = False
                node["required"] = list(node.get("properties", {}))
            for value in node.values():
                close(value)
        elif isinstance(node, list):
            for value in node:
                close(value)

    close(body)
    if strict_tools:
        body["properties"]["arguments"]["properties"]["scope"]["enum"] = ["company", "macro", None]
        body["properties"]["arguments"]["properties"]["top_n"].update(minimum=1, maximum=20)
    if final_action:
        body["properties"]["tool"]["enum"] = ["finish", "abstain"]
    if read_spans:
        # Bind citations to exact observations already returned by the read tool.
        # This prevents model-generated quote/offset variations without accepting
        # any new fact: domain and semantic verification still run afterwards.
        branches = []
        for span in read_spans:
            branches.append(
                {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        key: {
                            "type": "integer" if key in {"start", "end"} else "string",
                            "enum": [span[key]],
                        }
                        for key in ("evidence_id", "start", "end", "quote")
                    },
                    "required": ["evidence_id", "start", "end", "quote"],
                }
            )
        body["$defs"]["SourceSpan"] = {"anyOf": branches}
    return {"name": schema.__name__, "schema": body, "strict": True}
