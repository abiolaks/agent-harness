"""
Agent Harness — a minimal, production-pattern agent framework.

Implements the core ideas from Aditya Bhargava's talk:
"What if the harness mattered more than the model?"

Core concepts demonstrated:
 1. Tools as self-describing functions (typed, documented, discoverable)
 2. Reason → Act → Observe loop (the agent's execution cycle)
 3. Human-in-the-loop for dangerous operations
 4. Partial Function Application to constrain tools safely
 5. The harness as a first-class engineering artifact

No dependencies beyond Python stdlib + requests.
Swap `call_model()` to use any LLM provider.

Usage:
    python harness.py          # run the demo
    python harness.py --chat   # interactive chat mode
"""

from __future__ import annotations

import json
import os
import sys
import inspect
import functools
import textwrap
import traceback
from typing import Any, Callable
from dataclasses import dataclass, field


# ═══════════════════════════════════════════════════════════════════════
# PART 0: Model interface (plug in any LLM here)
# ═══════════════════════════════════════════════════════════════════════

def call_model(system_prompt: str, messages: list[dict], tools: list[dict]) -> dict:
    """
    Call an LLM and return its response.
    Swap this function to use any provider: Anthropic, OpenAI, Ollama, etc.

    Currently uses Anthropic Messages API via the OneCLI proxy.
    Set ANTHROPIC_API_KEY to any placeholder — the proxy handles auth.
    """
    try:
        import requests
    except ImportError:
        print("Install requests: pip install requests")
        sys.exit(1)

    api_key = os.environ.get("ANTHROPIC_API_KEY", "placeholder")
    base_url = os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com")

    # Build Anthropic-format messages
    anthropic_messages = []
    for m in messages:
        anthropic_messages.append({"role": m["role"], "content": m["content"]})

    body = {
        "model": os.environ.get("MODEL", "claude-sonnet-4-5-20250514"),
        "max_tokens": 1024,
        "system": system_prompt,
        "messages": anthropic_messages,
        "tools": tools if tools else None,
    }
    # Remove None values
    body = {k: v for k, v in body.items() if v is not None}

    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }

    resp = requests.post(f"{base_url}/v1/messages", headers=headers, json=body, timeout=60)

    if resp.status_code != 200:
        raise RuntimeError(f"API error {resp.status_code}: {resp.text[:500]}")

    data = resp.json()

    # Parse Anthropic response format
    content_blocks = data.get("content", [])
    text_response = ""
    tool_calls = []

    for block in content_blocks:
        if block["type"] == "text":
            text_response += block["text"]
        elif block["type"] == "tool_use":
            tool_calls.append({
                "id": block["id"],
                "name": block["name"],
                "input": block["input"],
            })

    return {
        "text": text_response.strip(),
        "tool_calls": tool_calls,
        "stop_reason": data.get("stop_reason", "end_turn"),
    }


# ═══════════════════════════════════════════════════════════════════════
# PART 1: Tool definitions — the "agency" contract
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class Tool:
    """A self-describing tool the agent can use."""
    name: str
    description: str
    parameters: dict          # JSON Schema for parameters
    fn: Callable              # the actual Python function
    dangerous: bool = False   # requires human approval?
    category: str = "general"


class ToolRegistry:
    """
    Collects and manages tools. Each tool is a Python function with
    a docstring and type hints — those become the tool's description
    and schema that the LLM reads.
    """

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(
        self,
        dangerous: bool = False,
        category: str = "general",
        constrain: dict | None = None,  # PFA: pre-constrain arguments
    ) -> Callable:
        """
        Decorator that turns a Python function into a registered tool.

        The function's docstring becomes the tool description.
        Type hints become the JSON Schema for parameters.

        KEY IDEA from the talk: Partial Function Application (PFA).
        `constrain` pre-binds arguments, making tools safer.
        Example:
            @tools.register(constrain={"allowed_paths": ["/tmp"]})
            def write_file(path: str, content: str): ...

        Now the agent CANNOT write outside /tmp, no matter what it asks.
        """
        def decorator(fn: Callable) -> Callable:
            # Build JSON Schema from type hints
            sig = inspect.signature(fn)
            properties = {}
            required = []

            for param_name, param in sig.parameters.items():
                if param_name in (constrain or {}):
                    continue  # pre-bound, not exposed to LLM

                param_type = "string"
                if param.annotation is not inspect.Parameter.empty:
                    ann = param.annotation
                    if ann is int:
                        param_type = "integer"
                    elif ann is float:
                        param_type = "number"
                    elif ann is bool:
                        param_type = "boolean"

                properties[param_name] = {
                    "type": param_type,
                    "description": f"Parameter: {param_name}",
                }

                if param.default is inspect.Parameter.empty:
                    required.append(param_name)

            # If constrained, wrap the function with pre-bound args
            wrapped_fn = fn
            if constrain:
                @functools.wraps(fn)
                def wrapped_fn(*args, **kwargs):
                    return fn(*args, **{**constrain, **kwargs})

            tool = Tool(
                name=fn.__name__,
                description=textwrap.dedent(fn.__doc__ or "").strip(),
                parameters={
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
                fn=wrapped_fn,
                dangerous=dangerous,
                category=category,
            )
            self._tools[fn.__name__] = tool
            return wrapped_fn

        return decorator

    def get_all(self) -> list[Tool]:
        return list(self._tools.values())

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def to_api_format(self) -> list[dict]:
        """Convert tools to Anthropic API format."""
        return [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.parameters,
            }
            for t in self._tools.values()
        ]


# ═══════════════════════════════════════════════════════════════════════
# PART 2: The Harness — Reason → Act → Observe
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class AgentStep:
    """One step in the agent's execution trace."""
    step_num: int
    reasoning: str       # what the model thought
    action: str          # tool name (or "respond")
    action_input: dict   # parameters passed
    observation: str     # what came back
    approved: bool | None = None  # None = no approval needed


@dataclass
class AgentRun:
    """Complete execution trace of an agent run."""
    task: str
    steps: list[AgentStep] = field(default_factory=list)
    final_answer: str = ""

    def print_trace(self):
        """Pretty-print the full execution trace — invaluable for debugging."""
        print("\n" + "=" * 60)
        print(f"TASK: {self.task}")
        print("=" * 60)
        for s in self.steps:
            approval = ""
            if s.approved is False:
                approval = " [BLOCKED by human]"
            elif s.approved is True:
                approval = " [Approved by human]"

            print(f"\n  Step {s.step_num}: {s.action}{approval}")
            print(f"  ── Reasoning: {s.reasoning[:120]}")
            if s.action_input:
                print(f"  ── Input: {json.dumps(s.action_input, indent=2)[:200]}")
            print(f"  ── Result: {s.observation[:200]}")
        print(f"\n  FINAL ANSWER: {self.final_answer}")
        print("=" * 60)


class Harness:
    """
    The agent harness — the "scaffolding around the model."

    This is where all the ideas come together:
    - The Reason → Act → Observe loop
    - Human-in-the-loop for dangerous tools
    - Step tracking for debugging
    - The harness doesn't call the model directly — it orchestrates
    """

    def __init__(
        self,
        tools: ToolRegistry,
        system_prompt: str | None = None,
        max_steps: int = 10,
        require_approval: Callable[[Tool, dict], bool] | None = None,
    ):
        self.tools = tools
        self.system_prompt = system_prompt or self._default_system_prompt()
        self.max_steps = max_steps
        self._approval_fn = require_approval or self._default_approval
        self.run: AgentRun | None = None

    @staticmethod
    def _default_system_prompt() -> str:
        return textwrap.dedent("""\
            You are a helpful AI assistant with access to tools.
            When given a task, think step by step, use tools when needed,
            and provide a clear final answer.

            Rules:
            - Break complex tasks into steps. Use one tool at a time.
            - When you have enough information, respond directly without tools.
            - Be honest when you don't know something.
            - Cite your sources when using information from tools.
        """)

    @staticmethod
    def _default_approval(tool: Tool, params: dict) -> bool:
        """Default: ask the human for any dangerous tool."""
        if not tool.dangerous:
            return True
        print(f"\n  ⚠️  AGENT wants to use: {tool.name}({json.dumps(params, indent=2)})")
        print(f"     This tool is marked DANGEROUS: {tool.description[:100]}")
        ans = input("     Approve? [y/N]: ").strip().lower()
        return ans == "y"

    def run_task(self, task: str, verbose: bool = True) -> AgentRun:
        """Execute a task through the Reason → Act → Observe loop."""

        self.run = AgentRun(task=task)
        messages = [{"role": "user", "content": task}]
        tool_schemas = self.tools.to_api_format()

        for step_num in range(1, self.max_steps + 1):
            if verbose:
                print(f"\n  ⟳ Step {step_num}...", end=" ", flush=True)

            # ── REASON: ask the model what to do ──
            response = call_model(
                system_prompt=self.system_prompt,
                messages=messages,
                tools=tool_schemas if tool_schemas else None,
            )

            # If the model responds with text (no tool call), it's done
            if not response["tool_calls"]:
                self.run.final_answer = response["text"]
                if verbose:
                    print("done.")
                break

            # ── ACT: execute the tool(s) the model chose ──
            for tc in response["tool_calls"]:
                tool = self.tools.get(tc["name"])
                if not tool:
                    observation = f"Error: unknown tool '{tc['name']}'"
                    approved = None
                else:
                    # ── HUMAN-IN-THE-LOOP check ──
                    approved = self._approval_fn(tool, tc["input"])
                    if not approved:
                        observation = (
                            f"BLOCKED: Human rejected the use of '{tc['name']}'."
                            f" Do NOT retry this tool. Explain why you needed it"
                            f" and ask the human what to do instead."
                        )
                    else:
                        # ── Execute the tool ──
                        try:
                            result = tool.fn(**tc["input"])
                            observation = str(result)
                        except Exception as e:
                            observation = f"Error: {e}\n{traceback.format_exc()[-300:]}"

                # ── OBSERVE: feed the result back to the model ──
                if verbose:
                    status = "✓" if approved else "✗"
                    print(f"[{status} {tc['name']}]", end=" ", flush=True)

                self.run.steps.append(AgentStep(
                    step_num=step_num,
                    reasoning=response["text"],
                    action=tc["name"],
                    action_input=tc["input"],
                    observation=observation,
                    approved=approved,
                ))

                # Add assistant + tool result to message history
                messages.append({
                    "role": "assistant",
                    "content": json.dumps([{
                        "type": "tool_use",
                        "id": tc["id"],
                        "name": tc["name"],
                        "input": tc["input"],
                    }]),
                })
                messages.append({
                    "role": "user",
                    "content": json.dumps([{
                        "type": "tool_result",
                        "tool_use_id": tc["id"],
                        "content": observation,
                    }]),
                })

        else:
            # Max steps reached without final answer
            self.run.final_answer = (
                f"Reached max steps ({self.max_steps}) without completing the task."
            )

        if verbose:
            print()
        return self.run


# ═══════════════════════════════════════════════════════════════════════
# PART 3: Demo — build and run an agent
# ═══════════════════════════════════════════════════════════════════════

def build_demo_agent() -> Harness:
    """
    Build a research assistant agent with:
    - A calculator tool (safe)
    - A file writer (DANGEROUS — needs human approval)
    - A file reader (safe, but constrained to /tmp)
    - A web search stub (safe)

    KEY IDEAS shown here:
    1. Tools are just Python functions with docstrings
    2. PFA: `write_file` is constrained so it can ONLY write to /tmp
    3. `write_file` is marked dangerous → human must approve
    """
    tools = ToolRegistry()

    # ── Safe tool: basic math ──
    @tools.register(category="utility")
    def calculate(expression: str) -> str:
        """
        Evaluate a mathematical expression.
        Supports: +, -, *, /, **, %, and parentheses.
        Example: calculate(expression="2 + 3 * 4")
        """
        # Safe eval: only allow math operations
        allowed = set("0123456789+-*/.() **%")
        cleaned = "".join(c for c in expression if c in allowed)
        if not cleaned:
            return "Error: no valid math expression found"
        try:
            result = eval(cleaned, {"__builtins__": {}}, {})
            return f"Result: {result}"
        except Exception as e:
            return f"Math error: {e}"

    # ── Dangerous tool: file writing ──
    # KEY IDEA: PFA constrains this to /tmp — the agent CAN'T write elsewhere
    @tools.register(
        dangerous=True,
        category="filesystem",
        constrain={"base_dir": "/tmp/agent-demo"},
    )
    def write_file(filename: str, content: str, base_dir: str = "/tmp") -> str:
        """
        Write content to a file. Only works within the allowed base directory.
        The agent cannot write outside this directory.
        """
        import os
        os.makedirs(base_dir, exist_ok=True)
        # Safety: resolve path and verify it's within base_dir
        filepath = os.path.realpath(os.path.join(base_dir, filename))
        if not filepath.startswith(os.path.realpath(base_dir)):
            return f"SECURITY BLOCK: path traversal detected. Denied writing to {filepath}"
        with open(filepath, "w") as f:
            f.write(content)
        return f"Wrote {len(content)} bytes to {filepath}"

    # ── Safe tool: file reading ──
    @tools.register(category="filesystem")
    def read_file(filename: str) -> str:
        """
        Read content from a file. Only works within /tmp/agent-demo.
        """
        import os
        base_dir = "/tmp/agent-demo"
        filepath = os.path.realpath(os.path.join(base_dir, filename))
        if not filepath.startswith(os.path.realpath(base_dir)):
            return "Error: cannot read outside /tmp/agent-demo"
        if not os.path.exists(filepath):
            return f"Error: file not found: {filename}"
        with open(filepath) as f:
            return f.read()

    # ── Safe tool: get current time ──
    @tools.register(category="utility")
    def get_current_time() -> str:
        """Get the current date and time."""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S %Z")

    # ── Safe tool: fetch a URL ──
    @tools.register(category="web")
    def fetch_url(url: str) -> str:
        """
        Fetch and summarize the content of a web page.
        Returns the page title and a text summary.
        """
        try:
            import requests
            resp = requests.get(url, timeout=10, headers={"User-Agent": "AgentHarness/1.0"})
            resp.raise_for_status()
            # Very basic extraction — just return first 1000 chars
            text = resp.text[:2000]
            return f"Fetched {url} (status {resp.status_code}). Content preview: {text[:500]}..."
        except Exception as e:
            return f"Failed to fetch {url}: {e}"

    # Build the harness
    harness = Harness(
        tools=tools,
        system_prompt=textwrap.dedent("""\
            You are a research assistant agent. You help users with tasks
            by using the tools available to you.

            Available tools:
            - calculate: do math
            - write_file: save results to a file (requires human approval)
            - read_file: read back saved files
            - get_current_time: check the current time
            - fetch_url: get web page content

            Guidelines:
            - Think step by step. Use one tool at a time.
            - For multi-step tasks, explain your plan first, then execute.
            - When writing files, clearly state what you're saving and why.
            - Provide a clear final answer summarizing what you found or did.
        """),
        max_steps=10,
    )

    return harness


# ═══════════════════════════════════════════════════════════════════════
# PART 4: Run
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("Agent Harness Demo — Reason → Act → Observe")
    print("=" * 60)
    print()
    print("This demo shows:")
    print("  1. Tools defined as Python functions with docstrings")
    print("  2. Reason → Act → Observe execution loop")
    print("  3. Human-in-the-loop for dangerous operations")
    print("  4. Partial Function Application (PFA) for safety")
    print()
    print("The agent has these tools:")
    print("  • calculate(expression) — do math")
    print("  • write_file(filename, content) — save to /tmp [DANGEROUS]")
    print("  • read_file(filename) — read from /tmp")
    print("  • get_current_time() — check the time")
    print("  • fetch_url(url) — get web content")
    print()

    if "--chat" in sys.argv:
        # Interactive mode
        harness = build_demo_agent()
        print("Chat mode. Type 'quit' to exit, 'trace' to see last run details.")
        print()
        while True:
            task = input("You: ").strip()
            if task.lower() in ("quit", "exit", "q"):
                break
            if task.lower() == "trace":
                if harness.run:
                    harness.run.print_trace()
                else:
                    print("(No run yet)")
                continue
            if not task:
                continue
            run = harness.run_task(task)
            print(f"\nAgent: {run.final_answer}")
    else:
        # Run a demo task
        harness = build_demo_agent()
        task = "Calculate 15% of 250, then write the result to a file called 'result.txt'."

        print(f"Demo task: {task}")
        print()
        print("─" * 60)

        run = harness.run_task(task)

        print("─" * 60)
        print()

        # Print the execution trace
        run.print_trace()
