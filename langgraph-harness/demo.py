"""
LangGraph Harness Demo — the SAME concepts as harness.py but using LangChain + LangGraph.

Run this to see how the Reason→Act→Observe loop looks when built with
a production-grade framework instead of from scratch.

Compare with the bare-Python version:
    python harness.py             # from-scratch implementation
    python langgraph-harness/demo.py  # LangGraph implementation

Both implement the same concepts from the talk. The LangGraph version gives you
built-in checkpointing, streaming, LangSmith tracing, and subgraph support.

Requires:
    pip install langchain langchain-anthropic langgraph
"""

import sys
import os

# Ensure we can import from the parent
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def demo_dry_run():
    """
    Demonstrate the LangGraph harness with simulated tools.
    No API key needed — uses mock responses.

    This mirrors the dry_run_demo.py example but shows how LangGraph
    implements each concept from the talk.
    """
    print("=" * 60)
    print("LangGraph Harness Demo")
    print("Talk concepts → LangGraph primitives")
    print("=" * 60)
    print()
    print("Mapping:")
    print("  Tools as typed fns    → @tool + Pydantic BaseModel")
    print("  PFA (constrained args)→ functools.partial")
    print("  Reason→Act→Observe   → StateGraph agent→tools→agent")
    print("  Guardrails            → pre/post nodes")
    print("  Sub-agents            → subgraphs")
    print("  Human-in-the-loop     → interrupt()")
    print("  Tracing               → LangSmith (auto)")
    print()
    print("=" * 60)
    print()

    # For a real demo, you'd need langchain installed.
    # This shows the code structure.
    code = '''\
from langgraph_harness.harness import LangGraphHarness
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from pydantic import BaseModel, Field

# ── 1. Tools as typed Pydantic models ──
class SearchInput(BaseModel):
    query: str = Field(description="What to search for")

@tool(args_schema=SearchInput)
def search_kb(query: str) -> str:
    """Search the knowledge base for information."""
    return f"Results for '{query}': ..."

class CalculateInput(BaseModel):
    expression: str = Field(description="Math expression to evaluate")

@tool(args_schema=CalculateInput)
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression."""
    return f"Result: {eval(expression)}"

# ── 2. Dangerous tool → triggers interrupt() ──
class NotifyInput(BaseModel):
    recipient: str
    message: str

@tool(args_schema=NotifyInput)
def send_notification(recipient: str, message: str) -> str:
    """Send notification. DANGEROUS."""
    return f"Sent to {recipient}"

# ── 3. Build the harness ──
harness = LangGraphHarness(
    model=ChatAnthropic(model="claude-sonnet-4-5-20250514"),
    tools=[search_kb, calculate, send_notification],
    system_prompt="You are a helpful assistant.",
    max_steps=10,
    dangerous_tools={"send_notification"},  # ← will trigger interrupt()
    enable_pii_redaction=True,    # ← input guardrail
    enable_keyword_block=True,    # ← input guardrail
    enable_secret_leak=True,      # ← output guardrail
)

# ── 4. Run ──
result = harness.invoke("Process a refund for $149.99")
print(harness.get_final_answer(result))

# ── 5. Stream (see reasoning live) ──
for event in harness.stream("What time is it?"):
    messages = event.get("messages", [])
    if messages:
        last = messages[-1]
        print(last.content)
'''
    print("Example code:")
    print(code)
    print()
    print("─" * 60)
    print()
    print("KEY DIFFERENCES from bare-Python version:")
    print()
    print("1. Pydantic → JSON Schema is AUTOMATIC")
    print("   No manual type→schema mapping. Pydantic does it.")
    print()
    print("2. interrupt() is BUILT-IN")
    print("   No custom approval loop. LangGraph pauses and waits.")
    print("   Resume with: graph.invoke(Command(resume=...), config)")
    print()
    print("3. Checkpointing is FREE")
    print("   MemorySaver stores state. Swap to SqliteSaver for")
    print("   persistence across restarts. Every step checkpointed.")
    print()
    print("4. LangSmith tracing is AUTOMATIC")
    print("   Set LANGCHAIN_TRACING_V2=true and every call,")
    print("   every tool execution, every guardrail is logged.")
    print()
    print("5. Sub-agents are SUBGRAPHS")
    print("   Compile a child StateGraph, add as node in parent.")
    print("   Each sub-agent has its own tools and memory.")
    print()
    print("WHEN TO USE WHICH:")
    print("  Bare Python: learning, no dependencies, full control")
    print("  LangGraph: production, need checkpointing/streaming/tracing")
    print("=" * 60)


if __name__ == "__main__":
    demo_dry_run()
