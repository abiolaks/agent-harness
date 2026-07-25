# Agent Harness — Where It Fits & How To Use It

## The big picture

Think of software as three layers:

```
┌──────────────────────────────────────────┐
│  APPLICATION                            │  ← your app, your business logic
│  "Book a flight"  "Summarize this doc"   │
├──────────────────────────────────────────┤
│  HARNESS  ← what Bhargava's talk is about│  ← the scaffolding around the LLM
│  Tools · Safety · Loop · State · Memory  │
├──────────────────────────────────────────┤
│  MODEL                                  │  ← the LLM itself
│  Claude · GPT · Gemini · open-source     │
└──────────────────────────────────────────┘
```

Most people start by wiring their app directly to the model. That works
for demos. It breaks in production. The harness is the layer that makes
LLM behavior **reliable, safe, and debuggable**.

## What the harness does

| Concern | Without harness | With harness |
|---------|----------------|--------------|
| Tool calling | Model calls anything, no guardrails | Typed, constrained, auditable |
| Safety | "Trust the prompt" | Deterministic enforcement, human approval gates |
| Debugging | "Why did it do that?" | Full execution trace, every step recorded |
| Recovery | Agent gets stuck, you start over | Pause/resume, retry, fallback |
| Observability | Black box | Every Tool call, reasoning, and result logged |

## The core loop: Reason → Act → Observe

```
                  ┌──────────┐
                  │   TASK   │  "Find the cheapest flight to London"
                  └────┬─────┘
                       │
            ┌──────────▼──────────┐
            │      REASON          │  LLM thinks: "I need to search flights"
            │  What should I do?   │
            └──────────┬──────────┘
                       │
            ┌──────────▼──────────┐
            │       ACT            │  Calls: search_flights(dest="London")
            │  Execute the tool    │  (with safety check, human approval if needed)
            └──────────┬──────────┘
                       │
            ┌──────────▼──────────┐
            │     OBSERVE          │  Result: "3 flights found, £200-£450"
            │  Feed result back    │  Feeds this back to model
            └──────────┬──────────┘
                       │
                  ┌────▼────┐
                  │  Done?  │── No ──→ REASON again
                  └────┬────┘
                       │ Yes
                  ┌────▼────┐
                  │  ANSWER │  "The cheapest is BA249 at £200"
                  └─────────┘
```

This loop IS the framework. Everything else — tool registries, safety
gates, execution traces — supports this loop.

## Key concept: Partial Function Application (PFA)

The most underrated idea from the talk:

```python
# WITHOUT PFA: the agent can write anywhere
@tools.register()
def write_file(path: str, content: str):
    ...

# WITH PFA: pre-bind the directory — the agent CAN'T go outside it
@tools.register(constrain={"base_dir": "/tmp/safe"})
def write_file(path: str, content: str, base_dir: str = "/tmp"):
    ...
```

The constrained argument is **never shown to the LLM**. It can't override it.
This is deterministic safety — the kind that doesn't depend on the model
"behaving."

## Where agent harnesses fit in real software

### Pattern 1: Internal tool (low risk)
```
User: "Summarize the Q3 report"
  → Harness → read_file("q3_report.pdf") → summarize → respond
```
Safe tools, no human approval needed. Runs autonomously.

### Pattern 2: Customer-facing agent (medium risk)
```
Customer: "Cancel my subscription"
  → Harness → lookup_account() → cancel_subscription() → respond
             ↑ triggers human-in-the-loop here
```
Business-critical actions need approval gates. The harness enforces this
regardless of what the model "wants" to do.

### Pattern 3: Autonomous ops agent (high risk)
```
Monitor: "Disk at 95%"
  → Harness → analyze_logs() → plan_action() → run_cleanup()
                                          ↑ human approves before execution
```
The harness is the safety boundary. Multiple approval stages.

## Running this demo

```bash
# Install dependency
pip install requests

# Set your API key (OneCLI handles auth, use any placeholder)
export ANTHROPIC_API_KEY=placeholder

# Run the demo task
python harness.py

# Or interactive chat mode
python harness.py --chat
```

## What to build next

1. **Add memory** — store past conversations so the agent remembers context
2. **Add a skill system** — reusable prompt + tool bundles for common tasks
3. **Add eval traces** — record every run and score the agent's decisions
4. **Add streaming** — see the agent's reasoning as it happens
5. **Swap the model** — replace `call_model()` with OpenAI/Ollama to compare

## The real lesson

The talk's title "What if the harness mattered more than the model?" means:

> A well-designed harness around a mediocre model beats a raw powerful model
> with no scaffolding. The harness is where the ENGINEERING happens.

The model is a component. The harness is the product.
