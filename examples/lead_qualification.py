"""
Lead Qualification Agent

A practical harness that evaluates inbound sales leads against
qualification criteria, enriches with company data, and decides
whether to route to sales or nurture — with human override.

Real business value: sales team only spends time on qualified leads.
"""

from harness import ToolRegistry, Harness
import textwrap
import json


def build_qualification_agent():
    tools = ToolRegistry()

    # Qualification scoring rubric
    SCORING = {
        "company_size": {
            "1-10": 1, "11-50": 2, "51-200": 3,
            "201-1000": 4, "1000+": 5
        },
        "budget_range": {
            "under_1k": 1, "1k_5k": 2, "5k_20k": 3,
            "20k_100k": 4, "100k_plus": 5
        },
        "timeline": {
            "6_months_plus": 1, "3-6_months": 2,
            "1-3_months": 3, "this_month": 4, "this_week": 5
        },
        "role": {
            "individual": 1, "manager": 2, "director": 3,
            "vp": 4, "c_suite": 5
        },
    }

    # In production: real CRM/API calls
    CRM_DB = []

    @tools.register(category="scoring")
    def score_lead(
        company_size: str,
        budget_range: str,
        timeline: str,
        role: str,
        industry: str,
        use_case: str,
    ) -> str:
        """
        Score a lead on qualification criteria (0-100 scale).
        Takes company size, budget, timeline, decision-maker role,
        industry, and stated use case. Returns a JSON scorecard.

        Score breakdown: company_size (25pts), budget (25pts),
        timeline (20pts), role (15pts), fit (15pts).
        """
        cs = SCORING["company_size"].get(company_size, 1)
        br = SCORING["budget_range"].get(budget_range, 1)
        tl = SCORING["timeline"].get(timeline, 1)
        rl = SCORING["role"].get(role, 1)

        # Normalize each to its max possible
        score = (
            (cs / 5) * 25 +
            (br / 5) * 25 +
            (tl / 5) * 20 +
            (rl / 5) * 15 +
            15  # base fit score (would be LLM-evaluated in production)
        )

        level = (
            "hot" if score >= 80 else
            "warm" if score >= 60 else
            "cool" if score >= 40 else
            "cold"
        )

        action = {
            "hot": "Route to sales NOW. Schedule call within 24h.",
            "warm": "Send case study + schedule call this week.",
            "cool": "Add to nurture sequence. Check back in 30 days.",
            "cold": "Add to newsletter. Revisit in 90 days.",
        }

        return json.dumps({
            "score": round(score),
            "level": level,
            "action": action[level],
            "breakdown": {
                "company_size": round((cs / 5) * 25),
                "budget": round((br / 5) * 25),
                "timeline": round((tl / 5) * 20),
                "role": round((rl / 5) * 15),
                "fit": 15,
            },
        })

    @tools.register(category="enrichment")
    def lookup_company(domain: str) -> str:
        """
        Look up company information from their domain.
        Returns: name, industry, size, location, and recent news.
        """
        # Simulated — in production, call Clearbit/Apollo/ZoomInfo API
        return json.dumps({
            "name": "Acme Logistics Ltd",
            "domain": domain,
            "industry": "Logistics & Supply Chain",
            "size": "51-200 employees",
            "revenue": "$10M-$50M",
            "location": "Lagos, Nigeria",
            "founded": 2019,
            "tech_stack": ["AWS", "Python", "PostgreSQL"],
            "recent_news": "Raised Series A ($5M) in March 2026. Expanding to Ghana.",
        })

    @tools.register(dangerous=True, category="routing")
    def add_to_crm(
        name: str,
        email: str,
        company: str,
        score: int,
        level: str,
        action: str,
    ) -> str:
        """
        Add or update this lead in the CRM with qualification data.
        DANGEROUS: modifies CRM records and may trigger automations.
        """
        record = {
            "name": name,
            "email": email,
            "company": company,
            "qualification_score": score,
            "level": level,
            "action": action,
            "status": "qualified" if level in ("hot", "warm") else "nurturing",
            "created_at": "2026-07-25T12:00:00Z",
        }
        CRM_DB.append(record)
        return json.dumps({
            "status": "created",
            "crm_id": f"LEAD-{len(CRM_DB):04d}",
            "record": record,
        })

    @tools.register(category="response")
    def generate_outreach(level: str, name: str, company: str, use_case: str) -> str:
        """
        Generate a personalized outreach draft based on lead quality.
        Hot leads get a direct call script. Warm gets a case study email.
        Cool/cold get a nurture email.
        """
        if level == "hot":
            return f"""OUTREACH DRAFT (Call Script):
Hi {name}, thanks for your interest in how we can help {company} with {use_case}.
I noticed {company} is scaling quickly and our solution typically saves teams
like yours 15-20 hours/week on {use_case}. Do you have 15 min this week to
see a quick demo tailored to {company}?"""

        elif level == "warm":
            return f"""OUTREACH DRAFT (Email):
Subject: How companies like {company} solve {use_case}
Hi {name}, great connecting! Based on what you shared about {use_case},
I thought you'd find this case study relevant — [link].
Would a 20-min walkthrough of how we handle {use_case} be useful?"""

        else:
            return f"""OUTREACH DRAFT (Nurture):
Subject: Resources for {use_case}
Hi {name}, thanks for your interest! While we explore if there's a fit,
here are some resources on {use_case} that might help: [blog post], [guide].
I'll check back in a month to see if anything changed."""  # noqa: E501

    harness = Harness(
        tools=tools,
        system_prompt=textwrap.dedent("""\
            You are a lead qualification agent. Your job is to:
            1. Enrich the lead with company data
            2. Score the lead on qualification criteria
            3. Generate appropriate outreach
            4. Add to CRM (requires human approval)

            Be honest in scoring — not every lead is hot.
            The goal is to save the sales team's time, not inflate scores.

            Always explain WHY a lead got their score.
        """),
        max_steps=5,
    )

    return harness


if __name__ == "__main__":
    print("=" * 60)
    print("Lead Qualification Agent — Example")
    print("=" * 60)

    harness = build_qualification_agent()

    # Simulate an inbound lead from a demo request form
    lead = {
        "name": "Chidi Okafor",
        "email": "chidi@acmelogistics.com",
        "company_domain": "acmelogistics.com",
        "company_size": "51-200",
        "budget_range": "20k_100k",
        "timeline": "1-3_months",
        "role": "director",
        "industry": "Logistics",
        "use_case": "Automating invoice processing and shipment tracking",
    }

    task = f"""
    Qualify this inbound lead:

    Name: {lead['name']}
    Email: {lead['email']}
    Company domain: {lead['company_domain']}
    Company size: {lead['company_size']}
    Budget range: {lead['budget_range']}
    Timeline: {lead['timeline']}
    Decision maker role: {lead['role']}
    Use case: {lead['use_case']}

    Workflow: lookup company → score lead → generate outreach → add to CRM.
    """

    print(f"\nLead: {lead['name']} from {lead['company_domain']}")
    print(f"Use case: {lead['use_case']}")
    print()

    run = harness.run_task(task)
    run.print_trace()
