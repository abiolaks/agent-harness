"""
LangGraph Harness — the same agent harness concepts, built with LangChain + LangGraph.

Every concept from Aditya Bhargava's talk, mapped to production-grade framework primitives:

Talk concept          →  LangGraph implementation
─────────────────────────────────────────────────────
Tools as typed fns    →  @tool decorator + Pydantic args
PFA (constrained args)→  functools.partial + tool factories
Reason→Act→Observe   →  StateGraph with agent/tools edges
Guardrails            →  pre/post nodes + Pydantic validators
Sub-agents            →  subgraphs + StateGraph nesting
Human-in-the-loop     →  interrupt() + Command(resume=...)
Tracing               →  LangSmith (automatic)

Usage:
    from langgraph_harness import build_harness, run_task

    harness = build_harness(tools=[search, calculate, notify])
    run = harness.invoke({"messages": [HumanMessage(content="...")]})
"""
