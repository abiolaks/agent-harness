# Agent Harness — The Scaffolding Around the Model

A production-pattern agent framework implementing the ideas from
Aditya Bhargava's talk: **"What if the harness mattered more than the model?"**

```
┌──────────────────────────────────────────┐
│  APPLICATION                             │  ← your app, your business logic
│  "Book a flight"  "Summarize this doc"   │
├──────────────────────────────────────────┤
│  HARNESS  ← what this repo is           │  ← the scaffolding around the LLM
│  Tools · Guardrails · Sub-agents · Trace │
├──────────────────────────────────────────┤
│  MODEL                                  │  ← the LLM itself
│  Claude · GPT · Gemini · open-source     │
└──────────────────────────────────────────┘
```

## Quick start

```bash
pip install requests

# Dry run — no API key, simulates the full loop
python examples/dry_run_demo.py

# Real model (via OneCLI proxy — any placeholder key works)
export ANTHROPIC_API_KEY=placeholder
python harness.py
```

## Package structure

```
agent-harness/
├── harness.py                  # backward-compat re-export + demo
├── harness/
│   ├── __init__.py             # clean public API
│   ├── tool_registry.py        # Tool + ToolRegistry (PFA, danger flagging)
│   ├── harness.py              # Harness — Reason→Act→Observe loop
│   ├── guardrails.py           # Input/Output guardrails, rate limiting, allowlists
│   ├── subagent.py             # SubAgent + SubAgentPool
│   ├── tracing.py              # Tracer — execution observability
│   └── model.py                # ModelProvider, AnthropicProvider, MockProvider
├── examples/
│   ├── dry_run_demo.py         # No-API-key demo of the full loop
│   ├── support_triage.py       # Customer support ticket triage
│   ├── lead_qualification.py   # Sales lead scoring + routing
│   └── deploy_verification.py  # Post-deploy health + log checks
└── README.md
```

## Core concepts

### 1. Tools as self-describing functions

```python
from harness import ToolRegistry

tools = ToolRegistry()

@tools.register(category="utility")
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression. Supports +, -, *, /, **, %."""
    ...

@tools.register(dangerous=True, category="filesystem", constrain={"base_dir": "/tmp"})
def write_file(filename: str, content: str, base_dir: str = "/tmp") -> str:
    """Write content to a file."""
    ...
```

The function's docstring becomes the LLM's tool description. Type hints become the JSON Schema. One source of truth.

### 2. Partial Function Application (PFA)

The most underrated idea from the talk:

```python
# PFA: pre-bind `base_dir` — the model NEVER sees it, CANNOT override it
@tools.register(constrain={"base_dir": "/tmp/safe"})
def write_file(filename: str, content: str, base_dir: str = "/tmp"):
    ...
```

This is **deterministic safety** — it doesn't depend on the model "behaving."

### 3. Reason → Act → Observe loop

```
TASK → REASON (model thinks) → ACT (tool executes) → OBSERVE (result fed back) → ...
                                                                                    ↓
                                                                                 ANSWER
```

Every agent follows this loop. The harness manages it, the model drives it.

### 4. Guardrails — deterministic safety

```python
from harness import GuardrailPipeline, KeywordBlocklist, ContentFilter, RateLimiter

pipeline = GuardrailPipeline.default()  # PII redaction, keyword blocking, secret detection

# Or build your own:
pipeline = (GuardrailPipeline()
    .add_input(KeywordBlocklist(keywords=["ignore previous instructions"]))
    .add_output(ContentFilter())
    .add_operational(RateLimiter(max_calls=100)))
```

Guardrails enforce rules REGARDLESS of what the model asks for. "Prompts are not security."

### 5. Sub-agents — specialized, composable

```python
from harness import SubAgent, SubAgentPool, ToolRegistry

# Each sub-agent has its own tools, prompt, and scope
research_tools = ToolRegistry()
# ... register research-specific tools ...

researcher = SubAgent(
    name="researcher",
    description="Search docs and the web for answers",
    system_prompt="You are a thorough researcher.",
    tools=research_tools,
)

pool = SubAgentPool()
pool.add(researcher)
# The parent harness sees sub-agents as tools it can delegate to
```

### 6. Human-in-the-loop

Tools marked `dangerous=True` require human approval before execution. The harness pauses, shows what the model wants to do, and waits — no action happens without consent.

### 7. Execution tracing

```python
run = harness.run_task("Process this refund request")
run.print_trace()

# Step 1: REASON → search_knowledge_base
# Step 2: ACT    → search_knowledge_base(query="refund policy")
# Step 3: OBSERVE← Refunds processed in 5-7 business days...
# Step 4: DONE   → end_turn
```

Every step recorded — reasoning, tool calls, inputs, outputs, guardrail decisions. Debuggable, not a black box.

## Where agent harnesses fit

### Internal tool (low risk)
```
User: "Summarize the Q3 report"
  → Harness → read_file("q3.pdf") → summarize → respond
```
Safe tools, autonomous.

### Customer-facing (medium risk)
```
Customer: "Cancel my subscription"
  → Harness → lookup_account() → cancel_subscription()
             ↑ human approves here
```
Dangerous actions gated behind human approval.

### Autonomous ops (high risk)
```
Monitor: "Disk at 95%"
  → Harness → analyze_logs() → plan_action() → run_cleanup()
                                          ↑ human approves
```
Multiple approval stages, full trace for audit.

## What to build from here

1. **Memory** — store past runs so agents remember context across sessions
2. **Eval harness** — record runs and score decisions; now you're doing AI engineering
3. **Streaming** — see reasoning as it happens
4. **Multi-agent workflows** — chain sub-agents: researcher → writer → reviewer
5. **Swap the model** — drop in OpenAI/Ollama via a new ModelProvider subclass

## The real lesson

> A well-designed harness around a mediocre model beats a raw powerful model
> with no scaffolding. The harness is where the ENGINEERING happens.

The model is a component. The harness is the product.
