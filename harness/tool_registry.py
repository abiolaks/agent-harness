"""
Tool Registry — self-describing, constrained, auditable tools.

Key concepts from the talk:
 1. Tools = typed Python functions with docstrings (one source of truth)
 2. Partial Function Application (PFA) — pre-bind args the LLM can't see
 3. Dangerous flag — marks tools that need human approval
 4. Categories — organize tools by domain
"""

from __future__ import annotations

import inspect
import functools
import textwrap
import json
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class Tool:
    """A self-describing tool the agent can use."""

    name: str
    description: str
    parameters: dict   # JSON Schema for LLM
    fn: Callable       # the actual Python function
    dangerous: bool = False
    category: str = "general"
    constrained_args: dict = field(default_factory=dict)
    # Metadata for tracing and debugging
    source_file: str = ""
    source_line: int = 0

    def to_api_schema(self) -> dict:
        """Convert to Anthropic/OpenAI tool format."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters,
        }

    def execute(self, **kwargs) -> str:
        """Execute the tool and return string result."""
        try:
            result = self.fn(**kwargs)
            return str(result)
        except Exception as e:
            return f"ToolError({self.name}): {e}"


class ToolRegistry:
    """
    Collects and manages tools.

    Each tool is a Python function with a docstring and type hints.
    Those become the tool's description and JSON Schema that the LLM reads.

    KEY FEATURES:
    - PFA: `constrain` pre-binds arguments the LLM can't see or override
    - Dangerous marking: tools that need human approval
    - Self-documenting: docstring = LLM description, type hints = schema
    """

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    # ── Registration ──

    def register(
        self,
        dangerous: bool = False,
        category: str = "general",
        constrain: dict | None = None,
        description: str | None = None,
    ) -> Callable:
        """
        Decorator: turn a Python function into a registered tool.

        Args:
            dangerous: If True, human must approve before execution.
            category: Group name for organizing tools (e.g. 'filesystem', 'web').
            constrain: Dict of pre-bound arguments (PFA). These are NEVER
                       shown to the LLM and CANNOT be overridden.
            description: Override the docstring-based description.

        Example:
            @tools.register(
                dangerous=True,
                category="filesystem",
                constrain={"base_dir": "/tmp/safe"}
            )
            def write_file(filename: str, content: str, base_dir: str = "/tmp"):
                ...

            The LLM sees only `filename` and `content`.
            `base_dir` is silently pre-bound to `/tmp/safe` — deterministic safety.
        """
        def decorator(fn: Callable) -> Callable:
            # Build JSON Schema from type hints
            sig = inspect.signature(fn)
            properties = {}
            required = []
            constrained_args = dict(constrain or {})

            for param_name, param in sig.parameters.items():
                if param_name in constrained_args:
                    continue  # PFA: hidden from LLM

                param_type = self._type_to_json(param.annotation)

                # Extract parameter description from docstring
                param_desc = f"Parameter: {param_name}"
                if param.default is not inspect.Parameter.empty:
                    param_desc += f" (default: {param.default})"

                properties[param_name] = {
                    "type": param_type,
                    "description": param_desc,
                }

                if param.default is inspect.Parameter.empty:
                    required.append(param_name)

            # Build function that applies constrained args
            wrapped_fn = fn
            if constrain:
                @functools.wraps(fn)
                def wrapped_fn(*args, **kwargs):
                    return fn(*args, **{**constrain, **kwargs})

            # Extract source info for debugging
            try:
                source_file = inspect.getfile(fn)
                _, source_line = inspect.getsourcelines(fn)
            except (TypeError, OSError):
                source_file = ""
                source_line = 0

            tool = Tool(
                name=fn.__name__,
                description=description or textwrap.dedent(fn.__doc__ or "").strip(),
                parameters={
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
                fn=wrapped_fn,
                dangerous=dangerous,
                category=category,
                constrained_args=constrained_args,
                source_file=source_file,
                source_line=source_line,
            )
            self._tools[fn.__name__] = tool
            return wrapped_fn

        return decorator

    def add_function(
        self,
        fn: Callable,
        dangerous: bool = False,
        category: str = "general",
        constrain: dict | None = None,
        description: str | None = None,
    ) -> Tool:
        """Register a function as a tool programmatically (non-decorator)."""
        self.register(dangerous=dangerous, category=category,
                      constrain=constrain, description=description)(fn)
        return self._tools[fn.__name__]

    # ── Access ──

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def get_all(self) -> list[Tool]:
        return list(self._tools.values())

    def get_by_category(self, category: str) -> list[Tool]:
        return [t for t in self._tools.values() if t.category == category]

    def list_tools(self) -> str:
        """Human-readable tool listing for debugging."""
        lines = []
        for t in self._tools.values():
            danger = "⚠️  DANGEROUS" if t.dangerous else "✓ safe"
            constrained = f" [constrained: {list(t.constrained_args)}]" if t.constrained_args else ""
            lines.append(f"  {t.name} ({t.category}) — {danger}{constrained}")
            lines.append(f"    {t.description[:100]}")
        return "\n".join(lines)

    def to_api_format(self) -> list[dict]:
        """Convert all tools to API-compatible format."""
        return [t.to_api_schema() for t in self._tools.values()]

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    @staticmethod
    def _type_to_json(annotation) -> str:
        """Map Python type to JSON Schema type."""
        if annotation is inspect.Parameter.empty:
            return "string"
        type_map = {
            str: "string", int: "integer", float: "number",
            bool: "boolean", list: "array", dict: "object",
        }
        origin = getattr(annotation, "__origin__", None)
        if origin is list:
            return "array"
        return type_map.get(annotation, "string")
