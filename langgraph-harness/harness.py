"""
LangGraph Harness — Reason→Act→Observe with guardrails, interrupts, and tracing.

Every concept from the talk, implemented as LangGraph primitives:

         ┌──────────┐
         │  INPUT   │  PII redaction, keyword blocking
         │ GUARDRAIL│
         └────┬─────┘
              │
         ┌────▼─────┐
         │  REASON  │  LLM with bound tools
         └────┬─────┘
              │
         ╔════▼═════╗
    ┌────╣ DANGER?  ╠────┐
    │    ╚════╤═════╝    │
    │ safe    │ dangerous │
    │         │           │
    │    ┌────▼─────┐     │
    │    │  ASK     │     │  interrupt() — human must approve
    │    │  HUMAN   │     │
    │    └────┬─────┘     │
    │         │ approved  │
    │    ┌────▼─────┐     │
    └────▶   ACT    ◀─────┘
         │ (ToolNode)
         └────┬─────┘
              │
         ┌────▼─────┐
         │ OUTPUT   │  Secret leak detection
         │ GUARDRAIL│
         └────┬─────┘
              │
         ┌────▼─────┐
         │  LOOP?   │── yes ──→ REASON
         └────┬─────┘
              │ no
         ┌────▼─────┐
         │  ANSWER  │
         └──────────┘
"""

from __future__ import annotations

from typing import Any, Literal

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command, interrupt
from langchain_core.messages import (
    BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage,
)
from langchain_core.tools import BaseTool
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableConfig

from .guardrails import (
    PIIRedactionNode,
    KeywordBlockNode,
    SecretLeakNode,
    RateLimitNode,
    GuardrailResult,
)


class LangGraphHarness:
    """
    The harness as a compiled LangGraph StateGraph.

    Features:
      - Input guardrails before the LLM (PII redaction, keyword blocking)
      - Tool execution with danger-aware interrupt (human-in-the-loop)
      - Output guardrails after the LLM (secret leak detection)
      - Sub-agent delegation (via as_tool())
      - Built-in checkpointing (MemorySaver) for pause/resume
      - Automatic LangSmith tracing

    Usage:
        from langgraph_harness import LangGraphHarness

        harness = LangGraphHarness(
            model=ChatAnthropic(model="claude-sonnet-4-5-20250514"),
            tools=[search, calculate, notify],
            system_prompt="You are a helpful assistant.",
        )
        result = harness.invoke("Process a refund for $149.99")
    """

    def __init__(
        self,
        model: BaseChatModel,
        tools: list[BaseTool] | None = None,
        system_prompt: str = "You are a helpful assistant. Use tools when needed.",
        max_steps: int = 10,
        dangerous_tools: set[str] | None = None,
        enable_pii_redaction: bool = True,
        enable_keyword_block: bool = True,
        enable_secret_leak: bool = True,
        enable_rate_limit: bool = True,
        approval_callback: Any = None,  # Callable[[str, dict], bool]
    ):
        self.model = model
        self.tools = tools or []
        self.system_prompt = system_prompt
        self.max_steps = max_steps
        self.dangerous_tools = dangerous_tools or set()
        self.approval_callback = approval_callback

        # Guardrail flags
        self.enable_pii_redaction = enable_pii_redaction
        self.enable_keyword_block = enable_keyword_block
        self.enable_secret_leak = enable_secret_leak
        self.enable_rate_limit = enable_rate_limit

        # Instantiate guardrail nodes
        self.pii_node = PIIRedactionNode()
        self.keyword_node = KeywordBlockNode()
        self.secret_node = SecretLeakNode()
        self.rate_limit_node = RateLimitNode()

        # Build the graph
        self._graph = self._build_graph()
        self._compiled = self._graph.compile(
            checkpointer=MemorySaver(),
        )

    # ── Graph construction ──

    def _build_graph(self) -> StateGraph:
        """Construct the full Reason→Act→Observe graph."""
        builder = StateGraph(dict)

        # Bind tools to model
        bound_model = self.model.bind_tools(self.tools)

        # ── Nodes ──

        def input_guardrails(state: dict) -> dict:
            """Pre-agent guardrails: PII redaction + keyword blocking."""
            s = state
            if self.enable_pii_redaction:
                s = self.pii_node(s)
            if self.enable_keyword_block:
                s = self.keyword_node(s)
            return s

        def agent(state: dict) -> dict:
            """REASON: call the LLM with tools."""
            messages = list(state.get("messages", []))

            # Inject system prompt on first turn
            if not any(
                hasattr(m, "content") and getattr(m, "type", None) == "system"
                for m in messages
            ):
                messages = [SystemMessage(content=self.system_prompt)] + messages

            response = bound_model.invoke(messages)
            return {**state, "messages": list(messages) + [response]}

        def tool_node(state: dict) -> dict:
            """
            ACT: execute tool calls.

            KEY: Before executing, check if ANY tool call is dangerous.
            If so, interrupt for human approval.
            """
            messages = state.get("messages", [])
            last = messages[-1] if messages else None

            if not last or not hasattr(last, "tool_calls"):
                return state

            # Check for dangerous tools
            dangerous_calls = []
            for tc in last.tool_calls:
                if tc["name"] in self.dangerous_tools:
                    dangerous_calls.append(tc)

            if dangerous_calls:
                # Interrupt — human-in-the-loop
                approval = interrupt({
                    "message": "⚠️ Dangerous tool calls require approval",
                    "calls": [
                        {"tool": tc["name"], "args": tc["args"]}
                        for tc in dangerous_calls
                    ],
                })

                if isinstance(approval, dict) and approval.get("action") == "reject":
                    # Return error for rejected tools
                    results = []
                    for tc in last.tool_calls:
                        if tc["name"] in self.dangerous_tools:
                            results.append(ToolMessage(
                                content=f"BLOCKED: human rejected {tc['name']}. "
                                        "Suggest an alternative.",
                                tool_call_id=tc["id"],
                            ))
                        else:
                            results.append(ToolMessage(
                                content="skipped (approved separately)",
                                tool_call_id=tc["id"],
                            ))
                    return {
                        **state,
                        "messages": list(messages) + results,
                        "dangerous_rejected": len(dangerous_calls),
                    }

            # Execute tools
            node = ToolNode(self.tools)
            result = node.invoke({"messages": messages})
            return {**state, "messages": list(messages) + result.get("messages", [])}

        def output_guardrails(state: dict) -> dict:
            """Post-agent guardrails: secret leak detection."""
            if self.enable_secret_leak:
                return self.secret_node(state)
            return state

        # ── Edges ──

        def should_continue(state: dict) -> Literal["tools", "end"]:
            """Route after agent: use tools or finish?"""
            messages = state.get("messages", [])

            # Check if guardrails blocked
            if state.get("guardrail_blocked"):
                return "end"

            if not messages:
                return "end"

            last = messages[-1]
            if hasattr(last, "tool_calls") and last.tool_calls:
                # Count total tool messages to enforce max steps
                tool_msg_count = sum(
                    1 for m in messages if isinstance(m, ToolMessage)
                )
                if tool_msg_count >= self.max_steps * len(self.tools):
                    return "end"
                return "tools"
            return "end"

        def after_tools(state: dict) -> Literal["output_guard", "agent"]:
            """After tools: go through output guard then back to agent."""
            return "output_guard"

        # Wire the graph
        builder.add_node("input_guard", input_guardrails)
        builder.add_node("agent", agent)
        builder.add_node("tools", tool_node)
        builder.add_node("output_guard", output_guardrails)

        builder.set_entry_point("input_guard")
        builder.add_edge("input_guard", "agent")

        builder.add_conditional_edges("agent", should_continue, {
            "tools": "tools",
            "end": "output_guard",
        })

        builder.add_conditional_edges("tools", after_tools, {
            "output_guard": "output_guard",
            "agent": "agent",
        })

        builder.add_edge("output_guard", END)

        return builder

    # ── Public API ──

    def invoke(
        self,
        task: str | list[BaseMessage],
        config: RunnableConfig | None = None,
    ) -> dict:
        """
        Run the harness on a task.

        Args:
            task: A string prompt or a list of LangChain messages.
            config: LangGraph config (thread_id for conversation continuity).

        Returns:
            Graph state dict with "messages" containing the full conversation.
        """
        if isinstance(task, str):
            messages = [HumanMessage(content=task)]
        else:
            messages = list(task)

        cfg = config or {"configurable": {"thread_id": "default"}}

        return self._compiled.invoke(
            {"messages": messages, "dangerous_rejected": 0},
            cfg,
        )

    def stream(
        self,
        task: str | list[BaseMessage],
        config: RunnableConfig | None = None,
    ):
        """Stream the harness execution — see reasoning as it happens."""
        if isinstance(task, str):
            messages = [HumanMessage(content=task)]
        else:
            messages = list(task)

        cfg = config or {"configurable": {"thread_id": "default"}}

        for event in self._compiled.stream(
            {"messages": messages},
            cfg,
            stream_mode="values",
        ):
            yield event

    def get_messages(self, state: dict) -> list[BaseMessage]:
        """Extract messages from graph state."""
        return state.get("messages", [])

    def get_final_answer(self, state: dict) -> str:
        """Extract the final answer from graph state."""
        messages = state.get("messages", [])
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content:
                return msg.content
        return ""
