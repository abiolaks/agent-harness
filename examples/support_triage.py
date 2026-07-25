"""
Customer Support Triage Agent

A practical harness that classifies incoming support tickets,
extracts key information, suggests priority, and routes to the
right team — all with human review before routing.

Real business value: reduces triage time from 5 min/ticket to 30 sec.
"""

from harness import ToolRegistry, Harness, AgentRun
import textwrap
import json


def build_triage_agent():
    tools = ToolRegistry()

    # ── Ticket database (simulated) ──
    TICKET_DB = {}  # In real life: your helpdesk API / DB

    SUPPORT_TEAMS = {
        "billing": "billing@company.com",
        "technical": "tech@company.com",
        "account": "accounts@company.com",
        "general": "support@company.com",
    }

    PRIORITY_SLA = {
        "critical": "15 minutes",
        "high": "1 hour",
        "medium": "4 hours",
        "low": "24 hours",
    }

    @tools.register(category="triage")
    def classify_ticket(subject: str, body: str) -> str:
        """
        Analyze a support ticket and return its category and priority.

        Categories: billing, technical, account, general
        Priorities: critical (system down), high (blocked), medium (issue), low (question)

        Returns a JSON object with category, priority, and reasoning.
        """
        # In production: this would be an LLM call.
        # Here we simulate with keyword matching for the demo.
        subject_lower = (subject + " " + body).lower()

        # Priority detection
        if any(w in subject_lower for w in ["down", "crash", "cannot access", "urgent", "broken"]):
            priority = "critical"
        elif any(w in subject_lower for w in ["blocked", "stuck", "error", "failing"]):
            priority = "high"
        elif any(w in subject_lower for w in ["question", "how to", "what is", "help with"]):
            priority = "low"
        else:
            priority = "medium"

        # Category detection
        if any(w in subject_lower for w in ["invoice", "payment", "billing", "charge", "refund", "price"]):
            category = "billing"
        elif any(w in subject_lower for w in ["bug", "error", "broken", "not working", "crash", "slow"]):
            category = "technical"
        elif any(w in subject_lower for w in ["login", "password", "account", "access", "permission"]):
            category = "account"
        else:
            category = "general"

        return json.dumps({
            "category": category,
            "priority": priority,
            "sla": PRIORITY_SLA[priority],
            "team_email": SUPPORT_TEAMS[category],
            "reasoning": f"Classified as {priority}/{category} based on keyword analysis."
        })

    @tools.register(category="triage")
    def lookup_customer_history(email: str) -> str:
        """
        Look up a customer's ticket history by email.
        Returns recent tickets, account status, and any open issues.
        """
        # Simulated — in production this hits your CRM/helpdesk API
        return json.dumps({
            "email": email,
            "account_status": "active",
            "plan": "business",
            "open_tickets": 2,
            "recent_tickets": [
                {"id": "TKT-1042", "subject": "Invoice discrepancy", "status": "open", "date": "2026-07-20"},
                {"id": "TKT-0987", "subject": "API rate limit question", "status": "resolved", "date": "2026-07-10"},
            ],
            "notes": "Customer prefers email communication. VIP account."
        })

    @tools.register(dangerous=True, category="routing")
    def route_ticket(
        ticket_id: str,
        category: str,
        priority: str,
        assigned_team: str,
        summary: str,
    ) -> str:
        """
        Assign the ticket to a team and notify them.
        DANGEROUS: actually sends notifications and modifies ticket state.

        Parameters are pre-filled from classification results.
        """
        TICKET_DB[ticket_id] = {
            "status": "routed",
            "category": category,
            "priority": priority,
            "team": assigned_team,
            "summary": summary,
            "routed_at": "2026-07-25T12:00:00Z",
        }
        return json.dumps({
            "status": "routed",
            "ticket_id": ticket_id,
            "assigned_to": assigned_team,
            "sla": PRIORITY_SLA.get(priority, "unknown"),
            "notification_sent": True,
        })

    @tools.register(category="triage")
    def generate_response_template(category: str, priority: str, customer_name: str) -> str:
        """
        Generate an initial response template for the customer.
        Returns a draft that a human agent can customize and send.
        """
        templates = {
            "critical": f"Hi {customer_name}, we've received your urgent report and our {category} team is on it. We'll update you within 15 minutes.",
            "high": f"Hi {customer_name}, thanks for reaching out. Our {category} team has been notified and will respond within 1 hour.",
            "medium": f"Hi {customer_name}, we've received your request. Our {category} team will look into this within 4 hours.",
            "low": f"Hi {customer_name}, thanks for your question! Our {category} team will get back to you within 24 hours.",
        }
        return templates.get(priority, templates["medium"])

    harness = Harness(
        tools=tools,
        system_prompt=textwrap.dedent("""\
            You are a support triage agent. Your job is to:
            1. Classify incoming tickets by category and priority
            2. Look up customer history for context
            3. Generate a response template
            4. Route the ticket (requires human approval)

            Workflow:
            - First, classify the ticket
            - Then, look up the customer's history
            - Generate a response template
            - Finally, route the ticket with all gathered information

            Be thorough but efficient. Every ticket should have:
            category, priority, customer context, and a draft response.
        """),
        max_steps=6,
    )

    return harness


# ── Run ──
if __name__ == "__main__":
    print("=" * 60)
    print("Support Triage Agent — Example")
    print("=" * 60)

    harness = build_triage_agent()

    # Simulate an incoming ticket
    ticket = {
        "id": "TKT-1051",
        "from": "jane@acmecorp.com",
        "subject": "Cannot access billing portal — urgent",
        "body": "Hi, I'm trying to download our July invoice but the billing portal shows a 500 error. We need this for our audit by EOD. Please help! — Jane (Acme Corp)",
    }

    task = f"""
    Process this support ticket:

    Ticket ID: {ticket['id']}
    From: {ticket['from']}
    Subject: {ticket['subject']}
    Body: {ticket['body']}

    Follow the workflow: classify → lookup history → generate response → route.
    """

    print(f"\nIncoming ticket: {ticket['subject']}")
    print(f"From: {ticket['from']}")
    print()

    run = harness.run_task(task)
    run.print_trace()
