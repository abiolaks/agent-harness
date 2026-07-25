"""
Model Provider — pluggable LLM interface.

Swap models without touching your harness logic.
Supports Anthropic, OpenAI, or a simulated mock for testing.
"""

from __future__ import annotations

import json
import os
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class ModelResponse:
    """Normalized model response, regardless of provider."""
    text: str = ""
    tool_calls: list[dict] = field(default_factory=list)
    stop_reason: str = "end_turn"
    usage: dict = field(default_factory=dict)


class ModelProvider(ABC):
    """Abstract model provider. Implement for each LLM API."""

    @abstractmethod
    def generate(
        self,
        system_prompt: str,
        messages: list[dict],
        tools: list[dict] | None = None,
        max_tokens: int = 1024,
    ) -> ModelResponse:
        """Send a request and return a normalized response."""
        ...


class AnthropicProvider(ModelProvider):
    """
    Anthropic Claude via Messages API.
    Uses OneCLI proxy — set ANTHROPIC_API_KEY to any placeholder.
    """

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        self.model = model or os.environ.get("MODEL", "claude-sonnet-4-5-20250514")
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "placeholder")
        self.base_url = base_url or os.environ.get(
            "ANTHROPIC_BASE_URL", "https://api.anthropic.com"
        )

    def generate(
        self,
        system_prompt: str,
        messages: list[dict],
        tools: list[dict] | None = None,
        max_tokens: int = 1024,
    ) -> ModelResponse:
        try:
            import requests
        except ImportError:
            raise RuntimeError("requests package required: pip install requests")

        anthropic_messages = [
            {"role": m["role"], "content": m["content"]} for m in messages
        ]

        body = {
            "model": self.model,
            "max_tokens": max_tokens,
            "system": system_prompt,
            "messages": anthropic_messages,
        }
        if tools:
            body["tools"] = tools

        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

        resp = requests.post(
            f"{self.base_url}/v1/messages",
            headers=headers,
            json=body,
            timeout=120,
        )

        if resp.status_code != 200:
            raise RuntimeError(f"API error {resp.status_code}: {resp.text[:500]}")

        data = resp.json()
        text_response = ""
        tool_calls = []

        for block in data.get("content", []):
            if block["type"] == "text":
                text_response += block["text"]
            elif block["type"] == "tool_use":
                tool_calls.append({
                    "id": block["id"],
                    "name": block["name"],
                    "input": block["input"],
                })

        return ModelResponse(
            text=text_response.strip(),
            tool_calls=tool_calls,
            stop_reason=data.get("stop_reason", "end_turn"),
            usage=data.get("usage", {}),
        )


class MockProvider(ModelProvider):
    """
    Simulated model for testing. Returns pre-programmed responses.
    No API key needed — use for demos and unit tests.
    """

    def __init__(self, responses: list[ModelResponse] | None = None):
        self.responses = responses or []
        self.call_count = 0
        self.calls: list[dict] = []  # record of all calls for inspection

    def generate(
        self,
        system_prompt: str,
        messages: list[dict],
        tools: list[dict] | None = None,
        max_tokens: int = 1024,
    ) -> ModelResponse:
        self.calls.append({
            "system_prompt": system_prompt,
            "last_message": messages[-1] if messages else None,
            "tool_count": len(tools or []),
        })

        if self.call_count < len(self.responses):
            resp = self.responses[self.call_count]
            self.call_count += 1
            return resp

        return ModelResponse(
            text="I have completed the task.",
            tool_calls=[],
            stop_reason="end_turn",
        )

    def add_response(self, text: str = "", tool_calls: list[dict] | None = None):
        """Queue a response for the next call."""
        self.responses.append(ModelResponse(
            text=text,
            tool_calls=tool_calls or [],
            stop_reason="tool_use" if tool_calls else "end_turn",
        ))
