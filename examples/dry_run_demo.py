"""
Dry-run demo: run the harness with a simulated model (no API key needed).

This shows the harness executing the Reason→Act→Observe loop
with a fake LLM so you can see the full pattern without any setup.

Once you understand the flow, swap in a real model via harness.py.
"""

from harness import ToolRegistry, Harness, Tool
import json


def build_demo_harness():
    """Same harness from harness.py, but with a simulated model."""
    tools = ToolRegistry()

    @tools.register(category="utility")
    def calculate(expression: str) -> str:
        """Evaluate a mathematical expression. Supports +, -, *, /, **, %."""
        allowed = set("0123456789+-*/.() **%")
        cleaned = "".join(c for c in expression if c in allowed)
        try:
            result = eval(cleaned, {"__builtins__": {}}, {})
            return f"Result: {result}"
        except Exception as e:
            return f"Error: {e}"

    @tools.register(category="utility")
    def search_knowledge_base(query: str) -> str:
        """Search internal docs for information about a topic."""
        kb = {
            "refund": "Refunds are processed within 5-7 business days. Must be requested within 30 days of purchase.",
            "pricing": "Business plan: $49/mo (up to 10 users). Enterprise: $199/mo (unlimited users). Annual billing saves 20%.",
            "api": "REST API at api.company.com/v2. Rate limit: 1000 req/min. Auth: Bearer token via /auth endpoint.",
            "sla": "Standard SLA: 99.9% uptime. Critical issues: 1h response. Enterprise: 99.99% with 15min response.",
        }
        query_lower = query.lower()
        for key, value in kb.items():
            if key in query_lower:
                return f"Found: {value}"
        return f"No results for '{query}'. Try: refund, pricing, api, sla."

    @tools.register(dangerous=True, category="action")
    def send_notification(recipient: str, message: str) -> str:
        """Send an email/Slack notification. DANGEROUS: actually sends."""
        return f"✓ Notification sent to {recipient}: {message[:80]}..."

    @tools.register(category="utility")
    def get_current_time() -> str:
        """Get the current date and time."""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    harness = Harness(
        tools=tools,
        max_steps=6,
    )
    return harness


# ── Simulate the model (no API key needed) ──
def simulate_model_calls(harness):
    """
    Monkey-patch call_model to return pre-planned responses,
    demonstrating the full Reason→Act→Observe loop.
    """
    import harness as hmod

    call_count = [0]
    responses = [
        # Step 1: Search KB for refund policy
        {
            "text": "I need to check the refund policy first.",
            "tool_calls": [{
                "id": "tool_001",
                "name": "search_knowledge_base",
                "input": {"query": "refund policy"},
            }],
            "stop_reason": "tool_use",
        },
        # Step 2: Calculate the refund amount
        {
            "text": "Now let me calculate 80% of $149.99.",
            "tool_calls": [{
                "id": "tool_002",
                "name": "calculate",
                "input": {"expression": "149.99 * 0.8"},
            }],
            "stop_reason": "tool_use",
        },
        # Step 3: Send notification (will trigger human approval)
        {
            "text": "I have all the information. Let me notify the customer.",
            "tool_calls": [{
                "id": "tool_003",
                "name": "send_notification",
                "input": {
                    "recipient": "customer@example.com",
                    "message": "Your refund of $119.99 has been approved and will be processed within 5-7 business days.",
                },
            }],
            "stop_reason": "tool_use",
        },
        # Step 4: Final answer
        {
            "text": "I've processed your request. Here's a summary:\n\n"
                    "• Refund policy: 5-7 business days, must be within 30 days of purchase ✓\n"
                    "• Refund amount: $119.99 (80% of $149.99)\n"
                    "• Notification sent to customer@example.com\n\n"
                    "The refund will be processed within 5-7 business days.",
            "tool_calls": [],
            "stop_reason": "end_turn",
        },
    ]

    def fake_call(system_prompt, messages, tools):
        if call_count[0] < len(responses):
            resp = responses[call_count[0]]
            call_count[0] += 1
            return resp
        return {"text": "Done.", "tool_calls": [], "stop_reason": "end_turn"}

    hmod.call_model = fake_call


if __name__ == "__main__":
    print("=" * 60)
    print("Agent Harness — Dry Run Demo (No API Key Needed)")
    print("=" * 60)
    print()
    print("This runs the full harness with a simulated LLM so you")
    print("can see the Reason→Act→Observe loop in action.")
    print()
    print("Task: A customer wants a refund for their $149.99 purchase.")
    print("The agent will: search KB → calculate refund → notify customer")
    print()

    harness = build_demo_harness()
    simulate_model_calls(harness)

    task = """
    A customer (customer@example.com) is requesting a refund for their
    $149.99 purchase. They're eligible for an 80% refund per our policy.
    Look up the refund policy, calculate the amount, and notify them.
    """

    run = harness.run_task(task)
    run.print_trace()

    print("\n─" * 60)
    print()
    print("WHAT JUST HAPPENED:")
    print("  1. REASON: agent decided to check refund policy first")
    print("  2. ACT: called search_knowledge_base('refund policy')")
    print("  3. OBSERVE: got policy details back")
    print("  4. REASON: agent decided to calculate the refund")
    print("  5. ACT: called calculate('149.99 * 0.8')")
    print("  6. OBSERVE: got $119.99")
    print("  7. REASON: agent decided to notify customer")
    print("  8. ACT: called send_notification() → ⚠️ triggers human approval")
    print("  9. OBSERVE: notification sent (if approved)")
    print(" 10. REASON: agent summarizes and finishes")
    print()
    print("This is the pattern. Every agent you build follows this loop.")
    print("The harness handles safety, tracing, and approval — you provide")
    print("the tools and the task.")
