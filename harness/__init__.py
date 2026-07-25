"""
Agent Harness — the scaffolding matters more than the model.

Core concepts from Aditya Bhargava's "What if the harness mattered
more than the model?" — implemented as a production-pattern library.

Usage:
    from harness import ToolRegistry, Harness, Guardrail, SubAgent

    tools = ToolRegistry()

    @tools.register(constrain={"base_dir": "/tmp"})
    def read_file(path: str, base_dir: str = "/tmp") -> str: ...

    harness = Harness(tools=tools)
    run = harness.run_task("Summarize the Q3 report")
    run.print_trace()
"""

from harness.tool_registry import Tool, ToolRegistry
from harness.harness import Harness, AgentStep, AgentRun
from harness.guardrails import (
    Guardrail,
    GuardrailResult,
    InputGuardrail,
    OutputGuardrail,
    ContentFilter,
    RateLimiter,
    ToolAllowlist,
    GuardrailPipeline,
)
from harness.subagent import SubAgent, SubAgentPool
from harness.tracing import Tracer, TraceLevel
from harness.model import ModelProvider, AnthropicProvider, MockProvider

__all__ = [
    # Core
    "Tool", "ToolRegistry",
    "Harness", "AgentStep", "AgentRun",
    # Guardrails
    "Guardrail", "GuardrailResult",
    "InputGuardrail", "OutputGuardrail",
    "ContentFilter", "RateLimiter", "ToolAllowlist",
    "GuardrailPipeline",
    # Sub-agents
    "SubAgent", "SubAgentPool",
    # Observability
    "Tracer", "TraceLevel",
    # Model
    "ModelProvider", "AnthropicProvider", "MockProvider",
]
