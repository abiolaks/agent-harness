"""
Tracing — execution observability for debugging agent loops.

From the talk: you need to see what happened, step by step.
Trace every model call, tool execution, and guardrail decision.
"""

from __future__ import annotations

import time
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class TraceLevel(str, Enum):
    """How much detail to trace."""
    OFF = "off"
    STEPS = "steps"          # one line per step
    DETAILED = "detailed"    # tool inputs and outputs
    FULL = "full"            # raw API messages


@dataclass
class TraceEvent:
    """A single event in the execution trace."""
    step: int
    timestamp: float
    event_type: str  # "reason", "act", "observe", "guardrail", "error", "complete"
    data: dict = field(default_factory=dict)

    @property
    def elapsed_ms(self) -> int:
        return int(self.data.get("elapsed_ms", 0))

    def summary(self) -> str:
        """One-line summary for quick scanning."""
        et = self.event_type
        if et == "reason":
            return f"  [{self.step:02d}] REASON → {self.data.get('tool', 'thinking')}"
        elif et == "act":
            return f"  [{self.step:02d}] ACT    → {self.data.get('tool', '?')}({self._short_input()})"
        elif et == "observe":
            result = self.data.get("result", "")
            short = result[:80].replace("\n", " ")
            return f"  [{self.step:02d}] OBSERVE← {short}"
        elif et == "guardrail":
            return f"  [{self.step:02d}] GUARD  → {self.data.get('reason', 'blocked')}"
        elif et == "error":
            return f"  [{self.step:02d}] ERROR  → {self.data.get('error', 'unknown')}"
        elif et == "complete":
            return f"  [{self.step:02d}] DONE   → {self.data.get('stop_reason', 'end')}"
        return f"  [{self.step:02d}] {et}: {self.data}"

    def _short_input(self) -> str:
        inp = self.data.get("input", {})
        if not inp:
            return ""
        items = [f"{k}={json.dumps(v)[:40]}" for k, v in inp.items()]
        return ", ".join(items)[:100]


@dataclass
class Tracer:
    """
    Records every step of the agent's execution loop.

    Usage:
        tracer = Tracer(level=TraceLevel.DETAILED)
        harness = Harness(tools=tools, tracer=tracer)
        harness.run_task("Do something")
        tracer.print_trace()
    """

    level: TraceLevel = TraceLevel.STEPS
    events: list[TraceEvent] = field(default_factory=list)
    start_time: float = 0.0

    def __post_init__(self):
        self.start_time = time.time()

    def record(self, step: int, event_type: str, **data) -> TraceEvent:
        """Record an event in the trace."""
        now = time.time()
        data["elapsed_ms"] = int((now - self.start_time) * 1000)
        event = TraceEvent(
            step=step,
            timestamp=now,
            event_type=event_type,
            data=data,
        )
        self.events.append(event)
        return event

    # ── Convenience methods ──

    def reason(self, step: int, text: str, tool: str = "") -> TraceEvent:
        if self.level == TraceLevel.OFF:
            return None
        return self.record(step, "reason", text=text[:200], tool=tool)

    def act(self, step: int, tool_name: str, tool_input: dict) -> TraceEvent:
        if self.level == TraceLevel.OFF:
            return None
        if self.level == TraceLevel.STEPS:
            return self.record(step, "act", tool=tool_name, input_summary=str(list(tool_input.keys())))
        return self.record(step, "act", tool=tool_name, input=tool_input)

    def observe(self, step: int, result: str) -> TraceEvent:
        if self.level == TraceLevel.OFF:
            return None
        return self.record(step, "observe", result=result[:500])

    def guardrail_block(self, step: int, reason: str, guardrail_name: str = "") -> TraceEvent:
        if self.level == TraceLevel.OFF:
            return None
        return self.record(step, "guardrail", reason=reason, name=guardrail_name, blocked=True)

    def error(self, step: int, error_msg: str, tool_name: str = "") -> TraceEvent:
        if self.level == TraceLevel.OFF:
            return None
        return self.record(step, "error", error=error_msg, tool=tool_name)

    def complete(self, step: int, stop_reason: str, answer_preview: str = "") -> TraceEvent:
        if self.level == TraceLevel.OFF:
            return None
        return self.record(step, "complete", stop_reason=stop_reason, preview=answer_preview[:200])

    # ── Output ──

    def print_trace(self) -> None:
        """Print a readable trace of the entire run."""
        if not self.events:
            print("  (no trace events)")
            return

        total_ms = self.events[-1].data.get("elapsed_ms", 0)
        print(f"\n  Trace ({len(self.events)} events, {total_ms}ms total)")
        print("  " + "─" * 58)
        for evt in self.events:
            print(evt.summary())

    def to_json(self) -> str:
        """Export trace as JSON for logging or analysis."""
        return json.dumps(
            [{"step": e.step, "type": e.event_type, "ms": e.data.get("elapsed_ms", 0), **e.data}
             for e in self.events],
            indent=2,
            default=str,
        )

    def clear(self) -> None:
        """Reset the trace."""
        self.events.clear()
        self.start_time = time.time()
