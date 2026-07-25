"""
Agent Harness — quick demo + backward-compat re-export.

For production code, import from the harness package directly:
    from harness import ToolRegistry, Harness, Tracer

This file exists so `python harness.py` still runs the demo and
old import paths still work.
"""

from harness.tool_registry import ToolRegistry, Tool
from harness.harness import Harness, AgentStep, AgentRun
from harness.guardrails import (
    GuardrailPipeline, GuardrailResult,
    InputGuardrail, OutputGuardrail,
    KeywordBlocklist, PIIFilter, MaxLengthGuardrail,
    ContentFilter, RateLimiter, ToolAllowlist,
)
from harness.subagent import SubAgent, SubAgentPool
from harness.tracing import Tracer, TraceLevel
from harness.model import ModelProvider, AnthropicProvider, MockProvider, ModelResponse

__all__ = [
    "ToolRegistry", "Tool",
    "Harness", "AgentStep", "AgentRun",
    "GuardrailPipeline", "GuardrailResult",
    "InputGuardrail", "OutputGuardrail",
    "KeywordBlocklist", "PIIFilter", "MaxLengthGuardrail",
    "ContentFilter", "RateLimiter", "ToolAllowlist",
    "SubAgent", "SubAgentPool",
    "Tracer", "TraceLevel",
    "ModelProvider", "AnthropicProvider", "MockProvider", "ModelResponse",
]


# ═══════════════════════════════════════════════════════════════════════
# Demo — build and run a sample agent
# ═══════════════════════════════════════════════════════════════════════

def build_demo_agent() -> Harness:
    """Build a research assistant with math, files, and web tools."""

    tools = ToolRegistry()

    @tools.register(category="utility")
    def calculate(expression: str) -> str:
        """
        Evaluate a mathematical expression.
        Supports: +, -, *, /, **, %, and parentheses.
        Example: calculate(expression="2 + 3 * 4")
        """
        allowed = set("0123456789+-*/.() **%")
        cleaned = "".join(c for c in expression if c in allowed)
        if not cleaned:
            return "Error: no valid math expression found"
        try:
            result = eval(cleaned, {"__builtins__": {}}, {})
            return f"Result: {result}"
        except Exception as e:
            return f"Math error: {e}"

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
        filepath = os.path.realpath(os.path.join(base_dir, filename))
        if not filepath.startswith(os.path.realpath(base_dir)):
            return f"SECURITY BLOCK: path traversal detected. Denied writing to {filepath}"
        with open(filepath, "w") as f:
            f.write(content)
        return f"Wrote {len(content)} bytes to {filepath}"

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

    @tools.register(category="utility")
    def get_current_time() -> str:
        """Get the current date and time."""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S %Z")

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
            text = resp.text[:2000]
            return f"Fetched {url} (status {resp.status_code}). Content preview: {text[:500]}..."
        except Exception as e:
            return f"Failed to fetch {url}: {e}"

    system_prompt = """\
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
    """

    return Harness(tools=tools, system_prompt=system_prompt, max_steps=10)


if __name__ == "__main__":
    import sys

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

    harness = build_demo_agent()

    if "--chat" in sys.argv:
        print("Chat mode. Type 'quit' to exit, 'trace' to see last run details.\n")
        run = None
        while True:
            task = input("You: ").strip()
            if task.lower() in ("quit", "exit", "q"):
                break
            if task.lower() == "trace":
                if run:
                    run.print_trace()
                else:
                    print("(No run yet)")
                continue
            if not task:
                continue
            run = harness.run_task(task)
            print(f"\nAgent: {run.final_answer}")
    else:
        task = "Calculate 15% of 250, then write the result to a file called 'result.txt'."
        print(f"Demo task: {task}\n")
        print("─" * 60)
        run = harness.run_task(task)
        print("─" * 60)
        run.print_trace()
