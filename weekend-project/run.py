"""
Expense Tracker Agent — Weekend Project

Combines BOTH talks from the AIE World's Fair:

Talk #1 (Aditya Bhargava/Etsy): Harness design
  → Tools, Reason→Act→Observe, guardrails, PFA

Talk #2 (Tisha Chawla & Susheem Koul/Microsoft): Production observability
  → Full tracing, structured logging, replay, failure reproduction

What you'll have by Sunday night:
  A working CLI agent that:
    1. Takes natural language expense descriptions
    2. Extracts structured data (vendor, amount, date, category)
    3. Saves to a ledger with guardrails (no duplicates, validation)
    4. Produces a FULL execution trace for every run
    5. Can replay any past run from its trace

Usage:
    python weekend_project/run.py "Lunch at Bukka with client — N15,000"
    python weekend_project/run.py --trace    # show last run's trace
    python weekend_project/run.py --replay 3 # replay run #3 from saved traces

NO API KEY NEEDED — uses MockProvider for the demo.
For real use: set ANTHROPIC_API_KEY=placeholder (OneCLI handles auth).
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json, csv, hashlib, textwrap
from datetime import datetime
from dataclasses import dataclass, field
from typing import Any

from harness import ToolRegistry, Harness, Tracer, TraceLevel
from harness import GuardrailPipeline, ContentFilter, RateLimiter
from harness.model import MockProvider, ModelResponse


# ═══════════════════════════════════════════════════════════════════════
# PART 1: The Ledger (your data store)
# ═══════════════════════════════════════════════════════════════════════

LEDGER_FILE = os.path.join(os.path.dirname(__file__), "ledger.csv")
TRACE_DIR = os.path.join(os.path.dirname(__file__), "traces")


def init_ledger():
    """Create the ledger file if it doesn't exist."""
    if not os.path.exists(LEDGER_FILE):
        with open(LEDGER_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "date", "vendor", "amount", "category", "description", "created_at"])


def read_ledger() -> list[dict]:
    """Read all entries from the ledger."""
    init_ledger()
    with open(LEDGER_FILE, newline="") as f:
        return list(csv.DictReader(f))


def entry_hash(vendor: str, amount: float, date: str) -> str:
    """Create a short hash to detect duplicates."""
    return hashlib.md5(f"{vendor.lower()}|{amount}|{date}".encode()).hexdigest()[:8]


# ═══════════════════════════════════════════════════════════════════════
# PART 2: Build the agent (combining both talks)
# ═══════════════════════════════════════════════════════════════════════

def build_expense_agent():
    tools = ToolRegistry()

    # ── Tool: Parse expense from natural language ──
    @tools.register(category="parsing")
    def extract_expense(text: str) -> str:
        """
        Extract structured expense data from natural language.
        Input: free-text description like "Lunch at Bukka with client — N15,000"
        Returns: JSON with vendor, amount_ngn, date, category, description.
        Categories: food, transport, software, office, client, learning, other.
        """
        today = datetime.now().strftime("%Y-%m-%d")
        return json.dumps({
            "vendor": "Bukka Restaurant",
            "amount_ngn": 15000,
            "date": today,
            "category": "food",
            "description": text,
        })

    # ── Tool: Check for duplicate ──
    @tools.register(category="validation")
    def check_duplicate(vendor: str, amount_ngn: int, date: str) -> str:
        """
        Check if this expense might be a duplicate entry.
        Compares vendor + amount + date against existing ledger entries.
        Returns: {"is_duplicate": bool, "matched_entry": {...} or null}
        """
        entries = read_ledger()
        h = entry_hash(vendor, float(amount_ngn), date)
        for e in entries:
            if entry_hash(e["vendor"], float(e["amount"]), e["date"]) == h:
                return json.dumps({"is_duplicate": True, "matched_entry": e})
        return json.dumps({"is_duplicate": False, "matched_entry": None})

    # ── Tool: Categorize (would call LLM in prod, keyword match for demo) ──
    @tools.register(category="classification")
    def categorize_expense(vendor: str, description: str) -> str:
        """
        Categorize an expense based on vendor and description.
        Categories: food, transport, software, office, client, learning, other.
        """
        desc = (vendor + " " + description).lower()
        if any(w in desc for w in ["food", "lunch", "dinner", "restaurant", "bukka", "cafe", "coffee", "breakfast"]):
            return json.dumps({"category": "food", "confidence": 0.9})
        if any(w in desc for w in ["uber", "bolt", "taxi", "transport", "fuel", "petrol", "flight", "bus"]):
            return json.dumps({"category": "transport", "confidence": 0.85})
        if any(w in desc for w in ["aws", "github", "api", "saas", "subscription", "hosting", "domain", "vscode", "cursor", "claude"]):
            return json.dumps({"category": "software", "confidence": 0.9})
        if any(w in desc for w in ["desk", "chair", "monitor", "printer", "paper", "office", "rent", "electricity"]):
            return json.dumps({"category": "office", "confidence": 0.85})
        if any(w in desc for w in ["client", "meeting", "proposal", "pitch", "dinner"]):
            return json.dumps({"category": "client", "confidence": 0.8})
        if any(w in desc for w in ["course", "book", "certification", "exam", "udemy", "coursera", "training", "workshop"]):
            return json.dumps({"category": "learning", "confidence": 0.9})
        return json.dumps({"category": "other", "confidence": 0.5})

    # ── Dangerous tool: Write to ledger ──
    @tools.register(dangerous=True, category="storage")
    def save_expense(
        vendor: str,
        amount_ngn: int,
        date: str,
        category: str,
        description: str,
    ) -> str:
        """
        Save a validated expense to the ledger. DANGEROUS: permanently records data.
        Only call after duplicate check and categorization.
        """
        init_ledger()
        entry_id = entry_hash(vendor, float(amount_ngn), date)
        with open(LEDGER_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                entry_id, date, vendor, amount_ngn, category, description,
                datetime.now().isoformat(),
            ])
        return json.dumps({
            "status": "saved",
            "id": entry_id,
            "vendor": vendor,
            "amount_ngn": amount_ngn,
            "category": category,
        })

    @tools.register(category="query")
    def get_summary(month: str = "") -> str:
        """
        Get expense summary: total by category. Month format: YYYY-MM.
        If no month given, returns current month.
        """
        if not month:
            month = datetime.now().strftime("%Y-%m")
        entries = read_ledger()
        month_entries = [e for e in entries if e["date"].startswith(month)]
        by_cat = {}
        total = 0
        for e in month_entries:
            amt = float(e["amount"])
            cat = e["category"]
            by_cat[cat] = by_cat.get(cat, 0) + amt
            total += amt
        return json.dumps({
            "month": month,
            "entry_count": len(month_entries),
            "total_ngn": total,
            "by_category": by_cat,
        })

    # ── Model: Simulated for demo (no API key) ──
    # This implements Talk #2: every model response is logged, tagged, replayable
    model = MockProvider(responses=[
        # Step 1: Extract
        ModelResponse(
            text="Let me parse this expense first.",
            tool_calls=[{
                "id": "t1", "name": "extract_expense",
                "input": {"text": "Lunch at Bukka with client — N15,000"},
            }],
            stop_reason="tool_use",
        ),
        # Step 2: Check duplicate
        ModelResponse(
            text="Now let me check if this is a duplicate.",
            tool_calls=[{
                "id": "t2", "name": "check_duplicate",
                "input": {"vendor": "Bukka Restaurant", "amount_ngn": 15000, "date": datetime.now().strftime("%Y-%m-%d")},
            }],
            stop_reason="tool_use",
        ),
        # Step 3: Categorize
        ModelResponse(
            text="Let me categorize this expense.",
            tool_calls=[{
                "id": "t3", "name": "categorize_expense",
                "input": {"vendor": "Bukka Restaurant", "description": "Lunch at Bukka with client"},
            }],
            stop_reason="tool_use",
        ),
        # Step 4: Save (dangerous — triggers approval)
        ModelResponse(
            text="Everything checks out. Saving to the ledger.",
            tool_calls=[{
                "id": "t4", "name": "save_expense",
                "input": {
                    "vendor": "Bukka Restaurant",
                    "amount_ngn": 15000,
                    "date": datetime.now().strftime("%Y-%m-%d"),
                    "category": "client",
                    "description": "Lunch at Bukka with client — N15,000",
                },
            }],
            stop_reason="tool_use",
        ),
        # Step 5: Final
        ModelResponse(
            text="✅ Expense saved!\n\n"
                 "• Vendor: Bukka Restaurant\n"
                 "• Amount: ₦15,000\n"
                 "• Category: client\n"
                 "• Status: recorded, not a duplicate\n\n"
                 "Run `--summary` to see this month's totals.",
            tool_calls=[],
            stop_reason="end_turn",
        ),
    ])

    # ── Guardrails: Talk #1 — protect the ledger ──
    guardrails = GuardrailPipeline.default()
    guardrails.add_operational(RateLimiter(max_calls=50, window_seconds=60))

    harness = Harness(
        tools=tools,
        model=model,
        system_prompt=textwrap.dedent("""\
            You are an expense tracking agent. Your job:
            1. Extract structured data from the user's expense description
            2. Check for duplicates against the ledger
            3. Categorize the expense
            4. Save to the ledger (requires approval)

            Always follow this flow: extract → check → categorize → save.
            Never skip validation. Never save without duplicate check.
            Amounts are in Nigerian Naira (₦).
        """),
        max_steps=6,
        guardrails=guardrails,
        tracer=Tracer(level=TraceLevel.DETAILED),
    )

    return harness


# ═══════════════════════════════════════════════════════════════════════
# PART 3: Observability layer (Talk #2)
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class TraceRecord:
    """A saved execution trace that can be replayed."""
    run_id: int
    timestamp: str
    task: str
    steps: list[dict]
    stop_reason: str
    final_answer: str
    total_ms: int
    model_calls: int

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "task": self.task,
            "steps": self.steps,
            "stop_reason": self.stop_reason,
            "final_answer": self.final_answer,
            "total_ms": self.total_ms,
            "model_calls": self.model_calls,
        }

    @classmethod
    def from_run(cls, run_id: int, task: str, run) -> "TraceRecord":
        return cls(
            run_id=run_id,
            timestamp=datetime.now().isoformat(),
            task=task,
            steps=[
                {
                    "step": s.step_number,
                    "reasoning": s.reasoning[:200],
                    "tool": s.tool_called,
                    "input": s.tool_input,
                    "result": s.tool_result[:300],
                    "dangerous": s.is_dangerous,
                    "approved": s.was_approved,
                    "blocked": s.guardrail_blocked,
                    "error": s.error,
                    "ms": s.elapsed_ms,
                }
                for s in run.steps
            ],
            stop_reason=run.stop_reason,
            final_answer=run.final_answer or "",
            total_ms=run.total_elapsed_ms,
            model_calls=len(run.steps),
        )


def save_trace(record: TraceRecord):
    """Persist a trace to disk for later replay."""
    os.makedirs(TRACE_DIR, exist_ok=True)
    path = os.path.join(TRACE_DIR, f"run-{record.run_id:04d}.json")
    with open(path, "w") as f:
        json.dump(record.to_dict(), f, indent=2)
    return path


def load_trace(run_id: int) -> TraceRecord | None:
    """Load a saved trace from disk."""
    path = os.path.join(TRACE_DIR, f"run-{run_id:04d}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    return TraceRecord(**data)


def replay_trace(run_id: int):
    """
    Reconstruct and display a past run from its trace.
    This IS the observability from Talk #2 — you can walk through
    exactly what the agent did, step by step, with timestamps.
    """
    record = load_trace(run_id)
    if not record:
        print(f"No trace found for run #{run_id}")
        return

    print(f"\n{'═' * 60}")
    print(f"REPLAY: Run #{record.run_id}")
    print(f"  Task: {record.task}")
    print(f"  When: {record.timestamp}")
    print(f"  Duration: {record.total_ms}ms ({record.model_calls} model calls)")
    print(f"  Outcome: {record.stop_reason}")
    print(f"{'═' * 60}")

    for s in record.steps:
        print(f"\n  ── Step {s['step']} ({s['ms']}ms) ──")
        print(f"  💭 {s['reasoning'][:150]}")
        if s['tool']:
            danger = "⚠️  DANGEROUS" if s['dangerous'] else "🔧"
            print(f"  {danger} {s['tool']}({json.dumps(s['input'])})")
        if s['result']:
            print(f"  → {s['result'][:200]}")
        if s['dangerous']:
            status = "✓ approved" if s['approved'] else "✗ rejected" if s['approved'] is not None else "? pending"
            print(f"  🛡 Approval: {status}")
        if s['blocked']:
            print(f"  🚫 GUARDRAIL BLOCKED")
        if s['error']:
            print(f"  ❌ {s['error']}")

    if record.final_answer:
        print(f"\n  Final answer:\n  {record.final_answer}")
    print(f"{'═' * 60}\n")


def list_traces() -> list[int]:
    """List all saved trace run IDs."""
    if not os.path.exists(TRACE_DIR):
        return []
    ids = []
    for fn in sorted(os.listdir(TRACE_DIR)):
        if fn.endswith(".json"):
            ids.append(int(fn.replace("run-", "").replace(".json", "")))
    return ids


def next_run_id() -> int:
    """Get the next run ID."""
    existing = list_traces()
    return max(existing) + 1 if existing else 1


# ═══════════════════════════════════════════════════════════════════════
# PART 4: Main entry point
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Expense Tracker Agent — AIE Talks #1 + #2 combined"
    )
    parser.add_argument(
        "description", nargs="?", type=str,
        help="Expense description (e.g. 'Lunch at Bukka — N15,000')"
    )
    parser.add_argument(
        "--trace", action="store_true",
        help="Show the last run's full execution trace"
    )
    parser.add_argument(
        "--replay", type=int, metavar="ID",
        help="Replay a saved run by ID"
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List all saved runs"
    )
    parser.add_argument(
        "--summary", action="store_true",
        help="Show expense summary for current month"
    )
    parser.add_argument(
        "--chat", action="store_true",
        help="Interactive mode"
    )
    parser.add_argument(
        "--auto-approve", action="store_true",
        help="Auto-approve dangerous operations (for demo/testing)"
    )

    args = parser.parse_args()

    # ── Info / replay modes (no agent needed) ──
    if args.list:
        ids = list_traces()
        if not ids:
            print("No saved traces yet. Run an expense first.")
        else:
            print(f"{len(ids)} saved runs:")
            for rid in ids:
                r = load_trace(rid)
                print(f"  #{rid}: {r.task[:80]} ({r.total_ms}ms, {r.stop_reason})")
        sys.exit(0)

    if args.replay:
        replay_trace(args.replay)
        sys.exit(0)

    if args.summary:
        harness = build_expense_agent()
        from harness.tool_registry import ToolRegistry as TR
        summary_tool = harness.tools.get("get_summary")
        if summary_tool:
            print(summary_tool.execute(
                month=datetime.now().strftime("%Y-%m")
            ))
        sys.exit(0)

    # ── Require a description for the main flow ──
    if not args.description and not args.chat:
        parser.print_help()
        print("\nExample: python run.py 'Lunch at Bukka with client — N15,000'")
        sys.exit(1)

    # ── Build and run ──
    init_ledger()
    harness = build_expense_agent()

    if args.auto_approve:
        harness.DEFAULT_APPROVAL_CONDITIONS["always_approve"] = True

    def run_one(desc: str) -> int:
        rid = next_run_id()
        print(f"\n  Processing: {desc}")
        print(f"  Run #{rid}\n")

        run = harness.run_task(desc)

        # Save trace (Talk #2: observability)
        record = TraceRecord.from_run(rid, desc, run)
        trace_path = save_trace(record)
        print(f"\n  Trace saved: {trace_path}")

        # Show result
        print(f"\n{run.final_answer}")
        return rid

    if args.chat:
        print("Expense Tracker — type expenses or 'summary', 'list', 'quit'")
        while True:
            cmd = input("\n> ").strip()
            if cmd.lower() in ("quit", "exit", "q"):
                break
            if cmd.lower() == "summary":
                s = harness.tools.get("get_summary").execute(
                    month=datetime.now().strftime("%Y-%m")
                )
                print(json.dumps(json.loads(s), indent=2))
                continue
            if cmd.lower() == "list":
                ids = list_traces()
                print(f"{len(ids)} saved runs" if ids else "No runs yet")
                continue
            if cmd.lower().startswith("replay"):
                parts = cmd.split()
                if len(parts) > 1:
                    replay_trace(int(parts[1]))
                else:
                    ids = list_traces()
                    if ids:
                        replay_trace(max(ids))
                continue
            if not cmd:
                continue
            run_one(cmd)
    else:
        run_one(args.description)

        # Also show the trace
        if args.trace:
            last_id = max(list_traces())
            replay_trace(last_id)
