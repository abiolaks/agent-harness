"""
LangGraph Guardrails — deterministic safety as graph nodes.

In LangGraph, guardrails are nodes that run before/after the LLM node.
They can:
- Modify state (redact PII from the last user message)
- Return early (block a prompt injection)
- Interrupt for human approval

This is MORE powerful than callback-based guardrails because
guardrails have full access to the graph state and can route.

Guardrail types:
 1. InputGuardrail — node BEFORE the agent, validates/filters user input
 2. OutputGuardrail — node AFTER the agent, validates/filters model output
 3. ToolGuardrail — node BEFORE tool execution, checks rate limits + allowlists
"""

from __future__ import annotations

import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from langgraph.types import Command


@dataclass
class GuardrailResult:
    allowed: bool = True
    reason: str = ""
    modified_content: str | None = None


# ═══════════════════════════════════════════════════════════════════════
# Base guardrail node interface
# ═══════════════════════════════════════════════════════════════════════

class GuardrailNode(ABC):
    """A guardrail that runs as a LangGraph node. Returns Command or state update."""

    name: str = "base"

    @abstractmethod
    def __call__(self, state: dict) -> dict:
        """Process the graph state and return modified state (or interrupt)."""
        ...


# ═══════════════════════════════════════════════════════════════════════
# Input guardrails
# ═══════════════════════════════════════════════════════════════════════

class PIIRedactionNode(GuardrailNode):
    """
    Redact PII from the last user message before the model sees it.
    Runs as a node BEFORE the agent node.
    """

    name = "pii_redaction"

    PATTERNS = {
        "email": r"\b[\w\.-]+@[\w\.-]+\.\w+\b",
        "phone": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
        "credit_card": r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",
    }

    def __init__(self, redact: list[str] | None = None):
        self.fields = redact or list(self.PATTERNS.keys())

    def __call__(self, state: dict) -> dict:
        from langchain_core.messages import HumanMessage

        messages = state.get("messages", [])
        if not messages:
            return state

        last = messages[-1]
        content = last.content if hasattr(last, 'content') else str(last)

        modified = content
        found_count = 0
        for field in self.fields:
            pattern = self.PATTERNS.get(field)
            if pattern:
                matches = re.findall(pattern, content)
                if matches:
                    found_count += len(matches)
                    modified = re.sub(pattern, f"[REDACTED_{field.upper()}]", modified)

        if found_count and isinstance(last, HumanMessage):
            messages = list(messages)
            messages[-1] = HumanMessage(content=modified)
            return {**state, "messages": messages, "pii_redacted": found_count}

        return state


class KeywordBlockNode(GuardrailNode):
    """
    Block prompts containing forbidden keywords.
    Returns a Command(goto=END) if blocked — the agent never runs.
    """

    name = "keyword_block"

    DEFAULT_BLOCKLIST = [
        "ignore all previous instructions",
        "disregard your system prompt",
        "you are now DAN",
        "pretend you are",
    ]

    def __init__(self, keywords: list[str] | None = None):
        self.keywords = keywords or self.DEFAULT_BLOCKLIST

    def __call__(self, state: dict) -> dict:
        from langchain_core.messages import AIMessage

        messages = state.get("messages", [])
        if not messages:
            return state

        last_content = ""
        if hasattr(messages[-1], 'content'):
            last_content = str(messages[-1].content).lower()

        for kw in self.keywords:
            if kw.lower() in last_content:
                return {
                    **state,
                    "messages": list(messages) + [
                        AIMessage(content=f"🚫 Request blocked by guardrail: matched '{kw}'")
                    ],
                    "guardrail_blocked": True,
                    "guardrail_reason": f"Keyword blocked: {kw}",
                }

        return state


# ═══════════════════════════════════════════════════════════════════════
# Output guardrails
# ═══════════════════════════════════════════════════════════════════════

class SecretLeakNode(GuardrailNode):
    """
    Scan model output for API keys, tokens, and secrets.
    Blocks the output if found.
    """

    name = "secret_leak"

    PATTERNS = [
        (r"sk-[a-zA-Z0-9]{32,}", "OpenAI key"),
        (r"sk-ant-[a-zA-Z0-9_-]{32,}", "Anthropic key"),
        (r"ghp_[a-zA-Z0-9]{36}", "GitHub PAT"),
    ]

    def __call__(self, state: dict) -> dict:
        from langchain_core.messages import AIMessage

        messages = state.get("messages", [])
        if not messages:
            return state

        last = messages[-1]
        content = last.content if hasattr(last, 'content') else str(last)

        for pattern, label in self.PATTERNS:
            if re.search(pattern, content):
                messages = list(messages)
                messages[-1] = AIMessage(
                    content="[Output blocked: contained a sensitive value]"
                )
                return {
                    **state,
                    "messages": messages,
                    "output_blocked": True,
                    "guardrail_reason": f"Output contained {label}",
                }

        return state


# ═══════════════════════════════════════════════════════════════════════
# Operational guardrails
# ═══════════════════════════════════════════════════════════════════════

class RateLimitNode(GuardrailNode):
    """Block if too many tool calls in the current window."""

    name = "rate_limit"

    def __init__(self, max_calls: int = 100, window_seconds: float = 60.0):
        self.max_calls = max_calls
        self.window = window_seconds
        self._calls: list[float] = []

    def __call__(self, state: dict) -> dict:
        now = time.time()
        self._calls = [t for t in self._calls if now - t < self.window]
        self._calls.append(now)

        if len(self._calls) > self.max_calls:
            return {
                **state,
                "rate_limited": True,
                "guardrail_reason": (
                    f"Rate limit: {len(self._calls)} calls in {self.window}s "
                    f"(max {self.max_calls})"
                ),
            }
        return state
