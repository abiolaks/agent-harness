"""
LangGraph Tools — self-describing, PFA-safe, danger-flagged.

Same concepts as harness/tool_registry.py but using LangChain's
@tool decorator + Pydantic for automatic JSON Schema generation.

Key difference from the bare-Python version:
LangChain does the type→schema mapping for you via Pydantic.

PFA is done with functools.partial — same idea, different syntax.
"""

from __future__ import annotations

import functools
from enum import Enum
from typing import Any, Callable

from pydantic import BaseModel, Field


class ToolCategory(str, Enum):
    UTILITY = "utility"
    FILESYSTEM = "filesystem"
    WEB = "web"
    DATABASE = "database"
    ACTION = "action"
    TRIAGE = "triage"
    SCORING = "scoring"
    ENRICHMENT = "enrichment"
    HEALTH = "health"
    LOGS = "logs"
    METRICS = "metrics"
    REPORTING = "reporting"
    ROUTING = "routing"


class ToolMeta(BaseModel):
    """Metadata attached to every tool — used by guardrails and the approval gate."""

    category: ToolCategory = ToolCategory.UTILITY
    dangerous: bool = False
    constrained_args: dict[str, Any] = Field(default_factory=dict)


def constrain(fn: Callable, **bound) -> Callable:
    """
    Partial Function Application — pre-bind arguments the LLM can't override.

    Usage:
        safe_write = constrain(write_file, base_dir="/tmp/safe")
        # LLM sees only: (filename: str, content: str)
        # base_dir is silently injected — deterministic safety.
    """
    return functools.partial(fn, **bound)
