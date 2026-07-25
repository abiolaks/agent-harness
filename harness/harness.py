"""
The Harness — the scaffolding that matters more than the model.

Core loop: Reason → Act → Observe → (repeat until done)

From the talk: the harness is the middleware between your app and the LLM.
It handles safety (guardrails), visibility (tracing), delegation (sub-agents),
and approval (human-in-the-loop) — all the things a raw API call doesn't.

Usage:
    tools = ToolRegistry()
    # ... register tools ...

    harness = Harness(
        tools=tools,
        model=AnthropicProvider(model="claude-sonnet-4-5-20250514"),
        system_prompt="You are a helpful assistant.",
        max_steps=10,
    )
    run = harness.run_task("Summarize Q3 sales")
    print(run.final_answer)
    run.print_trace()
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from harness.tool_registry import ToolRegistry, Tool
from harness.model import ModelProvider, ModelResponse, AnthropicProvider
from harness.guardrails import (
    GuardrailPipeline, GuardrailResult,
    ToolAllowlist, RateLimiter,
)
from harness.tracing import Tracer, TraceLevel
from harness.subagent import SubAgent, SubAgentPool


@dataclass
class AgentStep:
    """One step in the agent's execution trace."""
    step_number: int
    reasoning: str = ""
    tool_called: str = ""
    tool_input: dict = field(default_factory=dict)
    tool_result: str = ""
    is_dangerous: bool = False
    was_approved: bool | None = None  # None = not dangerous
    guardrail_blocked: bool = False
    guardrail_reason: str = ""
    elapsed_ms: int = 0
    error: str = ""


@dataclass
class AgentRun:
    """The complete result of running the harness."""
    task: str
    steps: list[AgentStep] = field(default_factory=list)
    final_answer: str | None = None
    stop_reason: str = "unknown"
    total_elapsed_ms: int = 0
    approved_dangerous_actions: int = 0
    blocked_actions: int = 0

    @property
    def last_text(self) -> str:
        """Last model text output (even if run didn't finish cleanly)."""
        return self.final_answer or ""

    def print_trace(self) -> None:
        """Print a human-readable execution trace."""
        print(f"\n  Agent Run Trace — {len(self.steps)} steps, "
              f"{self.total_elapsed_ms}ms total")
        print("  " + "═" * 60)
        for s in self.steps:
            print(f"\n  ── Step {s.step_number} ({s.elapsed_ms}ms) ──")
            if s.reasoning:
                print(f"  💭 REASON: {s.reasoning[:200]}")
            if s.tool_called:
                danger = "⚠️  DANGEROUS" if s.is_dangerous else "🔧"
                print(f"  {danger} ACT: {s.tool_called}({json.dumps(s.tool_input)})")
            if s.tool_result:
                result_preview = s.tool_result[:300].replace("\n", "\n    ")
                print(f"  👁  OBSERVE: {result_preview}")
            if s.is_dangerous and s.was_approved is not None:
                status = "✓ Approved" if s.was_approved else "✗ Rejected"
                print(f"  🛡  APPROVAL: {status}")
            if s.guardrail_blocked:
                print(f"  🚫 BLOCKED: {s.guardrail_reason}")
            if s.error:
                print(f"  ❌ ERROR: {s.error}")
        print(f"\n  Stop reason: {self.stop_reason}")
        if self.final_answer:
            print(f"\n  Final Answer:\n  {self.final_answer[:500]}")


class Harness:
    """
    The agent harness — the middleware layer between your app and the LLM.

    Responsibilities:
      - Reason→Act→Observe loop
      - Guardrail enforcement (input filtering, output filtering, rate limiting)
      - Human-in-the-loop for dangerous tool executions
      - Sub-agent dispatch
      - Execution tracing

    NOT responsible for:
      - Model calling (delegated to ModelProvider)
      - Tool definitions (delegated to ToolRegistry)
      - Message history management (you pass the conversation)
    """

    def __init__(
        self,
        tools: ToolRegistry,
        model: ModelProvider | None = None,
        system_prompt: str = "You are a helpful assistant. Use tools when needed.",
        max_steps: int = 10,
        guardrails: GuardrailPipeline | None = None,
        sub_agents: SubAgentPool | None = None,
        tracer: Tracer | None = None,
        # Human-in-the-loop callback: receives (tool_name, tool_input) → returns bool
        approval_callback: Callable[[str, dict], bool] | None = None,
    ):
        self.tools = tools
        self.model = model or AnthropicProvider()
        self.system_prompt = system_prompt
        self.max_steps = max_steps
        self.guardrails = guardrails or GuardrailPipeline.default()
        self.sub_agents = sub_agents or SubAgentPool()
        self.tracer = tracer or Tracer(level=TraceLevel.STEPS)
        self.approval_callback = approval_callback

    # ── Public API ──

    def run_task(
        self,
        task: str,
        messages: list[dict] | None = None,
    ) -> AgentRun:
        """
        Run a task through the Reason→Act→Observe loop.

        Args:
            task: The user's task description.
            messages: Optional conversation history (list of {"role":..., "content":...}).

        Returns:
            AgentRun with full trace and final answer.
        """
        run = AgentRun(task=task)
        start_time = time.time()

        # Build initial conversation
        conversation = (messages or []) + [{"role": "user", "content": task}]

        # Merge sub-agent schemas into the tool list the model sees
        tool_schemas = self.tools.to_api_format()
        tool_schemas.extend(self.sub_agents.to_tool_schemas())

        # Apply tool allowlist if present
        for gr in self.guardrails.operational_guardrails:
            if isinstance(gr, ToolAllowlist):
                tool_schemas = gr.filter_tools(tool_schemas)

        for step_idx in range(1, self.max_steps + 1):
            step_start = time.time()

            # ── Input guardrails ──
            last_msg = conversation[-1]["content"] if conversation else ""
            gr_result = self.guardrails.check_input(last_msg)
            if not gr_result.allowed:
                run.steps.append(AgentStep(
                    step_number=step_idx,
                    guardrail_blocked=True,
                    guardrail_reason=gr_result.reason,
                ))
                run.stop_reason = "guardrail_blocked_input"
                run.blocked_actions += 1
                break

            # ── Operational guardrails (rate limit) ──
            op_result = self.guardrails.check_operational()
            if not op_result.allowed:
                run.steps.append(AgentStep(
                    step_number=step_idx,
                    guardrail_blocked=True,
                    guardrail_reason=op_result.reason,
                ))
                run.stop_reason = "rate_limited"
                run.blocked_actions += 1
                break

            # ── REASON: Call the model ──
            self.tracer.reason(step_idx, "Calling model...")
            try:
                response = self.model.generate(
                    system_prompt=self.system_prompt,
                    messages=conversation,
                    tools=tool_schemas if tool_schemas else None,
                )
            except Exception as e:
                self.tracer.error(step_idx, str(e))
                run.steps.append(AgentStep(
                    step_number=step_idx,
                    error=str(e),
                ))
                run.stop_reason = "model_error"
                break

            current_step = AgentStep(
                step_number=step_idx,
                reasoning=response.text,
            )

            # ── No tool calls → agent is done ──
            if not response.tool_calls:
                # Output guardrails on final answer
                out_result = self.guardrails.check_output(response.text)
                if not out_result.allowed:
                    current_step.guardrail_blocked = True
                    current_step.guardrail_reason = out_result.reason
                    run.steps.append(current_step)
                    run.stop_reason = "guardrail_blocked_output"
                    run.blocked_actions += 1
                    self.tracer.guardrail_block(step_idx, out_result.reason)
                    break

                run.final_answer = response.text
                run.stop_reason = response.stop_reason
                current_step.elapsed_ms = int((time.time() - step_start) * 1000)
                run.steps.append(current_step)
                self.tracer.complete(step_idx, response.stop_reason, response.text[:200])
                break

            # ── Process tool calls ──
            tool_results = []
            for tc in response.tool_calls:
                tool_name = tc["name"]
                tool_input = tc.get("input", {})

                # Check if this is a sub-agent call
                sub_agent = self.sub_agents.get(tool_name)
                if sub_agent:
                    self.tracer.act(step_idx, f"sub:{tool_name}", tool_input)
                    try:
                        result = sub_agent.execute(
                            task=tool_input.get("task", ""),
                            model=self.model,
                        )
                        self.tracer.observe(step_idx, result)
                    except Exception as e:
                        result = f"SubAgentError({tool_name}): {e}"
                        self.tracer.error(step_idx, str(e), tool_name)
                    tool_results.append({
                        "tool_use_id": tc["id"],
                        "content": result,
                    })
                    current_step.tool_called = tool_name
                    current_step.tool_input = tool_input
                    current_step.tool_result = result[:500]
                    continue

                # Regular tool execution
                tool = self.tools.get(tool_name)
                self.tracer.act(step_idx, tool_name, tool_input)

                if not tool:
                    err = f"Unknown tool: '{tool_name}'"
                    self.tracer.error(step_idx, err)
                    tool_results.append({
                        "tool_use_id": tc["id"],
                        "content": f"Error: {err}",
                    })
                    continue

                # ── Danger check → human approval ──
                if tool.dangerous:
                    current_step.is_dangerous = True
                    approved = self._request_approval(tool_name, tool_input)
                    current_step.was_approved = approved
                    if approved:
                        run.approved_dangerous_actions += 1
                    else:
                        tool_results.append({
                            "tool_use_id": tc["id"],
                            "content": f"BLOCKED: human did not approve {tool_name}. "
                                       f"Explain why you need this and suggest an alternative.",
                        })
                        current_step.tool_result = "Blocked — not approved"
                        continue

                # ── Execute ──
                try:
                    result = tool.execute(**tool_input)
                    self.tracer.observe(step_idx, result)
                except Exception as e:
                    result = f"ToolError({tool_name}): {e}"
                    self.tracer.error(step_idx, str(e), tool_name)

                tool_results.append({
                    "tool_use_id": tc["id"],
                    "content": result,
                })

                current_step.tool_called = tool_name
                current_step.tool_input = tool_input
                current_step.tool_result = result[:500]

            # ── OBSERVE: Add tool results to conversation ──
            # Build assistant message with tool_use blocks
            assistant_content = []
            if response.text:
                assistant_content.append({"type": "text", "text": response.text})
            for tc in response.tool_calls:
                assistant_content.append({
                    "type": "tool_use",
                    "id": tc["id"],
                    "name": tc["name"],
                    "input": tc.get("input", {}),
                })

            conversation.append({"role": "assistant", "content": assistant_content})

            # Build tool result message
            user_content = []
            for tr in tool_results:
                user_content.append({
                    "type": "tool_result",
                    "tool_use_id": tr["tool_use_id"],
                    "content": tr["content"],
                })
            conversation.append({"role": "user", "content": user_content})

            current_step.elapsed_ms = int((time.time() - step_start) * 1000)
            run.steps.append(current_step)

        else:
            # Ran out of steps — get the last text as answer
            run.stop_reason = "max_steps_reached"

        run.total_elapsed_ms = int((time.time() - start_time) * 1000)
        return run

    # ── Approval ──

    DEFAULT_APPROVAL_CONDITIONS = {
        "always_approve": False,
        "auto_approve_tools": [],  # list of tool names
    }

    def _request_approval(self, tool_name: str, tool_input: dict) -> bool:
        """
        Request human approval for a dangerous tool execution.

        Returns True if approved, False if denied.
        """
        # Auto-approve if configured
        if tool_name in self.DEFAULT_APPROVAL_CONDITIONS.get("auto_approve_tools", []):
            return True

        if self.approval_callback:
            try:
                return self.approval_callback(tool_name, tool_input)
            except Exception:
                return False

        # Default: ask interactively
        print(f"\n  ╔════════════════════════════════════════╗")
        print(f"  ║  ⚠️  DANGEROUS ACTION                  ║")
        print(f"  ║  Tool: {tool_name:<32} ║")
        inp_summary = json.dumps(tool_input)[:40]
        print(f"  ║  Input: {inp_summary:<31} ║")
        print(f"  ╚════════════════════════════════════════╝")

        try:
            answer = input("  Approve? [y/N]: ").strip().lower()
            return answer in ("y", "yes")
        except (EOFError, KeyboardInterrupt):
            return False

    # ── Static helpers ──

    @staticmethod
    def run_sync(
        tools: ToolRegistry,
        task: str,
        system_prompt: str = "You are a helpful assistant.",
        max_steps: int = 5,
    ) -> AgentRun:
        """One-liner: create a harness, run a task, return the result."""
        harness = Harness(
            tools=tools,
            system_prompt=system_prompt,
            max_steps=max_steps,
        )
        return harness.run_task(task)
