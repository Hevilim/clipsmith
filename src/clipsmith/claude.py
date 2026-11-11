"""Calling Claude with a single "submit" tool and getting its input back."""

from __future__ import annotations

import os
from typing import Any

DEFAULT_MODEL = os.environ.get("CLIPSMITH_MODEL", "claude-sonnet-5-5")


def make_client() -> Any:
    import anthropic

    return anthropic.Anthropic()


def call_tool(client: Any, request: dict[str, Any], tool_name: str) -> dict[str, Any]:
    """Send `request` and return the input of the `tool_name` call.

    Newer models reject a forced tool_choice, so the tool is only offered ("auto") and
    the prompt asks for it; if the model answers in prose instead, ask once more.
    """
    request = {**request, "tool_choice": {"type": "auto"}}
    response = client.messages.create(**request)
    for _ in range(2):
        for block in response.content:
            if getattr(block, "type", None) == "tool_use" and block.name == tool_name:
                return block.input
        if getattr(response, "stop_reason", None) == "max_tokens":
            break
        request = {
            **request,
            "messages": [
                *request["messages"],
                {"role": "assistant", "content": response.content},
                {"role": "user", "content": f"Please submit your answer by calling `{tool_name}`."},
            ],
        }
        response = client.messages.create(**request)
    raise RuntimeError(f"Claude did not call `{tool_name}`.")
