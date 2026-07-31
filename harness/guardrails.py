"""
Guardrails — deterministic safety that doesn't depend on model behavior.

From the talk: "prompts are not security."
Guardrails enforce rules REGARDLESS of what the model asks for.

Types of guardrails:
 1. InputGuardrail — validate/filter what goes INTO the model
 2. OutputGuardrail — validate/filter what comes OUT of the model
 3. ContentFilter — block sensitive content
 4. RateLimiter — prevent abuse
 5. ToolAllowlist — restrict which tools are available
 6. GuardrailPipeline — compose multiple guardrails
"""

from __future__ import annotations

import re
import time
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class GuardrailResult:
    """Result of a guardrail check."""
    allowed: bool = True
    reason: str = ""
    modified_content: str | None = None  # if the guardrail rewrote the content
    metadata: dict = field(default_factory=dict)


class Guardrail(ABC):
    """Base class for all guardrails."""

    name: str = "base"

    @abstractmethod
    def check(self, content: str, context: dict | None = None) -> GuardrailResult:
        """Check content against this guardrail."""
        ...


# ═══════════════════════════════════════════════════════════════════════
# Input Guardrails — filter what goes into the model
# ═══════════════════════════════════════════════════════════════════════

class InputGuardrail(Guardrail):
    """Guardrail applied to user input before it reaches the model."""


class PIIFilter(InputGuardrail):
    """
    Detect and redact personally identifiable information from prompts.
    Email, phone numbers, credit cards, SSNs — stripped before the LLM sees them.
    """

    name = "pii_filter"

    PATTERNS = {
        "email": r"\b[\w\.-]+@[\w\.-]+\.\w+\b",
        "phone": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
        "credit_card": r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",
        "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
    }

    def __init__(self, redact: list[str] | None = None):
        self.redact_fields = redact or list(self.PATTERNS.keys())

    def check(self, content: str, context: dict | None = None) -> GuardrailResult:
        modified = content
        found = []
        for field in self.redact_fields:
            pattern = self.PATTERNS.get(field)
            if pattern:
                matches = re.findall(pattern, content)
                if matches:
                    found.extend(matches)
                    modified = re.sub(pattern, f"[REDACTED_{field.upper()}]", modified)

        if found:
            return GuardrailResult(
                allowed=True,  # still allow, just redacted
                reason=f"Redacted {len(found)} PII instances: {', '.join(self.redact_fields)}",
                modified_content=modified,
                metadata={"redacted_count": len(found), "types": list(set(self.redact_fields))},
            )
        return GuardrailResult(allowed=True)


class KeywordBlocklist(InputGuardrail):
    """
    Block prompts containing forbidden keywords.
    Useful for preventing prompt injection, abuse, or off-topic requests.
    """

    name = "keyword_blocklist"

    def __init__(self, keywords: list[str] | None = None):
        self.keywords = keywords or [
            "ignore all previous instructions",
            "disregard your system prompt",
            "you are now DAN",
            "pretend you are",
        ]

    def check(self, content: str, context: dict | None = None) -> GuardrailResult:
        content_lower = content.lower()
        for kw in self.keywords:
            if kw.lower() in content_lower:
                return GuardrailResult(
                    allowed=False,
                    reason=f"Blocked by keyword: '{kw}'",
                    metadata={"matched_keyword": kw},
                )
        return GuardrailResult(allowed=True)


class MaxLengthGuardrail(InputGuardrail):
    """Reject prompts that exceed a maximum length."""

    name = "max_length"

    def __init__(self, max_chars: int = 32000):
        self.max_chars = max_chars

    def check(self, content: str, context: dict | None = None) -> GuardrailResult:
        if len(content) > self.max_chars:
            return GuardrailResult(
                allowed=False,
                reason=f"Input too long ({len(content)} chars, max {self.max_chars})",
            )
        return GuardrailResult(allowed=True)


# ═══════════════════════════════════════════════════════════════════════
# Output Guardrails — filter what comes out of the model
# ═══════════════════════════════════════════════════════════════════════

class OutputGuardrail(Guardrail):
    """Guardrail applied to model output before it reaches the user."""


class ContentFilter(OutputGuardrail):
    """
    Block model outputs containing specific patterns.
    Default: block API keys, secrets, and sensitive patterns.
    """

    name = "content_filter"

    SECRET_PATTERNS = [
        (r"sk-[a-zA-Z0-9]{32,}", "OpenAI API key"),
        (r"sk-ant-[a-zA-Z0-9_-]{32,}", "Anthropic API key"),
        (r"AIza[0-9A-Za-z\-_]{35}", "Google API key"),
        (r"ghp_[a-zA-Z0-9]{36}", "GitHub personal access token"),
        (r"-----BEGIN (?:RSA |EC |DSA )?PRIVATE KEY-----", "Private key"),
    ]

    def __init__(self, extra_patterns: list[tuple[str, str]] | None = None):
        self.patterns = self.SECRET_PATTERNS + (extra_patterns or [])

    def check(self, content: str, context: dict | None = None) -> GuardrailResult:
        for pattern, label in self.patterns:
            if re.search(pattern, content):
                return GuardrailResult(
                    allowed=False,
                    reason=f"Output blocked: contains {label}",
                    metadata={"pattern": label},
                )
        return GuardrailResult(allowed=True)


# ═══════════════════════════════════════════════════════════════════════
# Operational Guardrails
# ═══════════════════════════════════════════════════════════════════════

class RateLimiter(Guardrail):
    """
    Rate limit tool executions to prevent abuse.
    Tracks calls per time window.
    """

    name = "rate_limiter"

    def __init__(self, max_calls: int = 100, window_seconds: float = 60.0):
        self.max_calls = max_calls
        self.window = window_seconds
        self._calls: list[float] = []

    def check(self, content: str = "", context: dict | None = None) -> GuardrailResult:
        now = time.time()
        # Purge old entries
        self._calls = [t for t in self._calls if now - t < self.window]
        self._calls.append(now)

        if len(self._calls) > self.max_calls:
            return GuardrailResult(
                allowed=False,
                reason=f"Rate limit exceeded: {len(self._calls)} calls in {self.window}s (max {self.max_calls})",
                metadata={"current_rate": len(self._calls), "limit": self.max_calls},
            )
        return GuardrailResult(allowed=True)


class ToolAllowlist(Guardrail):
    """
    Restrict which tools are available to an agent.
    Applied at the harness level — the model never sees blocked tools.

    This is PFA at the harness level: even if the model generates
    a tool call, it won't find it in the registry.
    """

    name = "tool_allowlist"

    def __init__(self, allowed_tools: list[str]):
        self.allowed = set(allowed_tools)

    def check(self, tool_name: str, context: dict | None = None) -> GuardrailResult:
        if tool_name not in self.allowed:
            return GuardrailResult(
                allowed=False,
                reason=f"Tool '{tool_name}' not in allowlist: {sorted(self.allowed)}",
            )
        return GuardrailResult(allowed=True)

    def filter_tools(self, tools: list[dict]) -> list[dict]:
        """Remove disallowed tools from the API schema."""
        return [t for t in tools if t["name"] in self.allowed]


# ═══════════════════════════════════════════════════════════════════════
# Pipeline — compose multiple guardrails
# ═══════════════════════════════════════════════════════════════════════

class GuardrailPipeline:
    """
    Chain multiple guardrails together.
    Applies input guardrails → model → output guardrails.
    Stops at the first rejection (fail-fast).
    """

    def __init__(self):
        self.input_guardrails: list[InputGuardrail] = []
        self.output_guardrails: list[OutputGuardrail] = []
        self.operational_guardrails: list[Guardrail] = []

    def add_input(self, guardrail: InputGuardrail) -> "GuardrailPipeline":
        self.input_guardrails.append(guardrail)
        return self

    def add_output(self, guardrail: OutputGuardrail) -> "GuardrailPipeline":
        self.output_guardrails.append(guardrail)
        return self

    def add_operational(self, guardrail: Guardrail) -> "GuardrailPipeline":
        self.operational_guardrails.append(guardrail)
        return self

    @staticmethod
    def _extract_text(content) -> str:
        """Extract plain text from a message content, which may be a string
        or a list of Anthropic content blocks."""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text", ""))
                elif isinstance(block, str):
                    parts.append(block)
            return "\n".join(parts)
        return str(content)

    @staticmethod
    def _is_tool_result(content) -> bool:
        """Check if content is a tool_result block (not user text)."""
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    return True
        return False

    def check_input(self, content, context: dict | None = None) -> GuardrailResult:
        """Run all input guardrails. Returns first rejection or last result."""
        # Skip guardrails on tool_result blocks — those aren't user input
        if self._is_tool_result(content):
            return GuardrailResult(allowed=True)
        modified = self._extract_text(content)
        for gr in self.input_guardrails:
            result = gr.check(modified, context)
            if not result.allowed:
                return result
            if result.modified_content is not None:
                modified = result.modified_content
        return GuardrailResult(allowed=True, modified_content=modified)

    def check_output(self, content, context: dict | None = None) -> GuardrailResult:
        """Run all output guardrails. Returns first rejection."""
        text = self._extract_text(content)
        for gr in self.output_guardrails:
            result = gr.check(text, context)
            if not result.allowed:
                return result
        return GuardrailResult(allowed=True)

    def check_operational(self, tool_name: str = "", context: dict | None = None) -> GuardrailResult:
        """Run operational guardrails (rate limit, tool allowlist)."""
        for gr in self.operational_guardrails:
            if isinstance(gr, ToolAllowlist):
                result = gr.check(tool_name, context)
            else:
                result = gr.check("", context)
            if not result.allowed:
                return result
        return GuardrailResult(allowed=True)

    @classmethod
    def default(cls) -> "GuardrailPipeline":
        """Create a pipeline with sensible defaults."""
        return (
            cls()
            .add_input(KeywordBlocklist())
            .add_input(MaxLengthGuardrail())
            .add_input(PIIFilter(redact=["email", "credit_card"]))
            .add_output(ContentFilter())
            .add_operational(RateLimiter(max_calls=100, window_seconds=60))
        )
