"""
LangGraph Sub-Agents — specialized agents as composable subgraphs.

In LangGraph, a sub-agent is a compiled StateGraph that:
  1. Has its own tools, system prompt, and model
  2. Is invoked as a node in the parent graph
  3. Returns a structured result the parent can use

The parent's tool-calling node sees sub-agents as regular tools —
same interface, but behind the scenes they run their own Reason→Act→Observe loop.

Key concept from the talk: "Don't make one giant agent. Make small,
focused agents that each do one thing well."
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import BaseTool
from langchain_core.language_models import BaseChatModel


@dataclass
class SubAgentConfig:
    """Configuration for a sub-agent."""

    name: str
    description: str
    system_prompt: str
    tools: list[BaseTool]
    max_steps: int = 5


class SubAgent:
    """
    A focused sub-agent that runs as a compiled LangGraph.

    Usage:
        researcher = SubAgent(SubAgentConfig(
            name="researcher",
            description="Search docs and web for answers",
            system_prompt="You are a thorough researcher...",
            tools=[search_tool, fetch_tool],
        ))

        # Register as a tool in the parent harness:
        parent_tools.append(researcher.as_tool())
    """

    def __init__(self, config: SubAgentConfig, model: BaseChatModel | None = None):
        self.config = config
        self.model = model
        self._graph = None

    def build_graph(self, model: BaseChatModel) -> StateGraph:
        """Build the Reason→Act→Observe graph for this sub-agent."""
        self.model = model
        bound_model = model.bind_tools(self.config.tools)

        def reason(state: dict) -> dict:
            """Agent node: call the LLM with the sub-agent's system prompt."""
            messages = state.get("messages", [])
            # Prepend system message on first call
            if not any(
                hasattr(m, "type") and m.type == "system" for m in messages
            ):
                from langchain_core.messages import SystemMessage
                messages = [SystemMessage(content=self.config.system_prompt)] + list(messages)

            response = bound_model.invoke(messages)
            return {**state, "messages": list(messages) + [response]}

        def should_continue(state: dict) -> str:
            """Route: tools or end?"""
            messages = state.get("messages", [])
            if not messages:
                return "end"
            last = messages[-1]
            if hasattr(last, "tool_calls") and last.tool_calls:
                # Count steps
                tool_msgs = [m for m in messages if isinstance(m, ToolMessage)]
                if len(tool_msgs) >= self.config.max_steps * len(self.config.tools):
                    return "end"
                return "tools"
            return "end"

        # Build graph
        builder = StateGraph(dict)
        builder.add_node("agent", reason)
        builder.add_node("tools", ToolNode(self.config.tools))
        builder.set_entry_point("agent")
        builder.add_conditional_edges("agent", should_continue, {
            "tools": "tools",
            "end": "__end__",
        })
        builder.add_edge("tools", "agent")

        self._graph = builder.compile(checkpointer=MemorySaver())
        return self._graph

    def invoke(self, task: str, model: BaseChatModel | None = None) -> str:
        """Run this sub-agent on a task and return its final answer."""
        m = model or self.model
        if m is None:
            raise ValueError("Model required — pass to SubAgent or invoke()")

        graph = self._graph or self.build_graph(m)

        config = {"configurable": {"thread_id": f"sub_{self.config.name}"}}
        result = graph.invoke(
            {"messages": [HumanMessage(content=task)]},
            config,
        )

        messages = result.get("messages", [])
        # Find the last AIMessage (final answer)
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content:
                return msg.content

        return f"[{self.config.name}] completed."

    def as_tool(self) -> BaseTool:
        """Expose this sub-agent as a tool the parent can call."""
        from langchain_core.tools import tool

        sub = self  # capture for closure

        @tool(name=sub.config.name, description=sub.config.description)
        def subagent_task(task: str) -> str:
            """Execute a task using this specialized sub-agent."""
            return sub.invoke(task)

        return subagent_task


@dataclass
class SubAgentPool:
    """Manage multiple sub-agents as a team."""

    def __init__(self):
        self._agents: dict[str, SubAgent] = {}

    def add(self, agent: SubAgent) -> "SubAgentPool":
        self._agents[agent.config.name] = agent
        return self

    def get(self, name: str) -> SubAgent | None:
        return self._agents.get(name)

    def as_tools(self) -> list[BaseTool]:
        """Return all sub-agents as tools for the parent graph."""
        return [agent.as_tool() for agent in self._agents.values()]

    def dispatch(self, name: str, task: str, model: BaseChatModel) -> str:
        """Run a named sub-agent on a task."""
        agent = self._agents.get(name)
        if not agent:
            return f"Error: no sub-agent '{name}'. Available: {list(self._agents)}"
        return agent.invoke(task, model=model)
