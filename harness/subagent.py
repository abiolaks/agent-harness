"""
Sub-agents — specialized, modular agents with their own tool sets.

From the talk: "Don't make one giant agent that does everything.
Make small, focused agents that each do one thing well."

A sub-agent:
 1. Has its own tools (scoped to its domain)
 2. Has its own system prompt (specialized role)
 3. Is invoked by the parent harness as a tool
 4. Returns structured results

Pattern: The parent agent delegates to sub-agents via tool calls.
The sub-agent runs its own Reason→Act→Observe loop and returns.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

from harness.tool_registry import ToolRegistry
from harness.model import ModelProvider, ModelResponse
from harness.tracing import Tracer


@dataclass
class SubAgent:
    """
    A focused sub-agent with its own tools and personality.

    Think of it as a specialist you can call as a tool.
    The parent agent sees it as a function in its tool list.

    Example:
        researcher = SubAgent(
            name="researcher",
            description="Search the knowledge base and web for information",
            system_prompt="You are a research assistant. Find accurate answers.",
            tools=research_tools,
        )
    """

    name: str
    description: str
    system_prompt: str
    tools: ToolRegistry
    model: ModelProvider | None = None  # inherits from parent if None
    max_steps: int = 5
    tracer: Tracer | None = None

    def to_tool_schema(self) -> dict:
        """Generate a tool schema so the parent can call this sub-agent."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": f"Task for the {self.name} sub-agent to perform",
                    },
                },
                "required": ["task"],
            },
        }

    def execute(self, task: str, model: ModelProvider | None = None) -> str:
        """
        Run this sub-agent on a task and return its final answer.

        This is called by the parent harness when the model invokes
        this sub-agent as a tool.
        """
        from harness.harness import Harness

        sub_harness = Harness(
            tools=self.tools,
            model=model or self.model,
            system_prompt=self.system_prompt,
            max_steps=self.max_steps,
            tracer=self.tracer,
        )
        run = sub_harness.run_task(task)
        return run.final_answer or run.last_text or f"[{self.name}] completed with no output."

    def __repr__(self) -> str:
        tool_count = len(self.tools)
        return f"SubAgent({self.name!r}, {tool_count} tools, max_steps={self.max_steps})"


@dataclass
class SubAgentPool:
    """
    Manage multiple sub-agents as a team.

    Each sub-agent is registered and exposed to the parent as a tool.
    The parent model can decide which specialist to call for each subtask.

    Example:
        pool = SubAgentPool()
        pool.add(researcher)
        pool.add(code_reviewer)
        pool.add(test_runner)

        # Register all as tools on the parent harness
        for schema in pool.to_tool_schemas():
            parent_tools.add_tool_schema(schema)

        # Execute a sub-agent by name
        result = pool.dispatch("researcher", "Find the latest docs on FastAPI")
    """

    def __init__(self):
        self._agents: dict[str, SubAgent] = {}

    def add(self, agent: SubAgent) -> "SubAgentPool":
        self._agents[agent.name] = agent
        return self

    def remove(self, name: str) -> "SubAgentPool":
        self._agents.pop(name, None)
        return self

    def get(self, name: str) -> SubAgent | None:
        return self._agents.get(name)

    def list_names(self) -> list[str]:
        return sorted(self._agents.keys())

    def to_tool_schemas(self) -> list[dict]:
        """Return tool schemas for all sub-agents (for parent's tool list)."""
        return [agent.to_tool_schema() for agent in self._agents.values()]

    def dispatch(self, name: str, task: str, model: ModelProvider | None = None) -> str:
        """Run a named sub-agent on a task."""
        agent = self._agents.get(name)
        if not agent:
            available = ", ".join(self.list_names())
            return f"Error: no sub-agent '{name}'. Available: {available}"
        return agent.execute(task, model=model)

    def __len__(self) -> int:
        return len(self._agents)

    def __repr__(self) -> str:
        return f"SubAgentPool({self.list_names()})"
