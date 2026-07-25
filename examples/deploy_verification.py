"""
Deploy Verification Agent

Runs a series of checks after a deployment: health endpoints,
error log scanning, key metric validation. Escalates to a human
if anything looks wrong.

Real business value: catches deployment issues before customers do.
Replaces the "deploy and pray" workflow with structured verification.
"""

from harness import ToolRegistry, AgentRun, Harness
import textwrap
import json
import random


def build_deploy_agent():
    tools = ToolRegistry()

    # Simulated service state
    SERVICE_STATE = {
        "api": {"healthy": True, "latency_ms": 45, "error_rate": 0.001},
        "worker": {"healthy": True, "queue_depth": 12, "processing_rate": 340},
        "database": {"healthy": True, "replication_lag_ms": 50, "connections": 18},
        "frontend": {"healthy": True, "bundle_size_kb": 842, "load_time_ms": 1200},
    }

    @tools.register(category="health")
    def check_health(service_name: str) -> str:
        """
        Hit the /health endpoint of a service.
        Returns: healthy/unhealthy status and key metrics.
        Services: api, worker, database, frontend
        """
        svc = SERVICE_STATE.get(service_name)
        if not svc:
            return json.dumps({"error": f"Unknown service: {service_name}"})

        # Simulate a random 15% chance of degraded state
        if random.random() < 0.15:
            svc["healthy"] = False
            svc["error_rate"] = 0.05  # spike

        return json.dumps({
            "service": service_name,
            "healthy": svc["healthy"],
            "metrics": svc,
        })

    @tools.register(category="logs")
    def scan_error_logs(service_name: str, minutes: int = 5) -> str:
        """
        Scan recent error logs for a service.
        Returns count and sample of errors since deploy.
        """
        # Simulated — in production, query your logging system (Datadog, Grafana, etc.)
        if random.random() < 0.1:
            return json.dumps({
                "service": service_name,
                "errors_since_deploy": 3,
                "samples": [
                    {"level": "ERROR", "message": "Connection timeout to payment gateway", "count": 2},
                    {"level": "WARNING", "message": "Slow query detected (2.3s)", "count": 1},
                ],
                "trend": "increasing" if random.random() < 0.3 else "stable",
            })
        return json.dumps({
            "service": service_name,
            "errors_since_deploy": 0,
            "samples": [],
            "trend": "clean",
        })

    @tools.register(category="metrics")
    def check_key_metrics(service_name: str) -> str:
        """
        Compare current metrics against baseline.
        Flags anything outside normal range.
        """
        baseline = {
            "api": {"latency_p95_ms": 100, "error_rate": 0.01},
            "worker": {"queue_depth_max": 50, "processing_rate_min": 200},
            "database": {"replication_lag_max_ms": 200, "connections_max": 50},
            "frontend": {"load_time_p95_ms": 3000, "bundle_size_max_kb": 1000},
        }

        current = SERVICE_STATE.get(service_name, {})
        bl = baseline.get(service_name, {})

        alerts = []
        for metric, values in current.items():
            if isinstance(values, (int, float)):
                threshold_key = f"{metric}_max"
                if threshold_key in bl and values > bl[threshold_key]:
                    alerts.append({
                        "metric": metric,
                        "current": values,
                        "threshold": bl[threshold_key],
                        "status": "BREACHED",
                    })

        return json.dumps({
            "service": service_name,
            "alerts": alerts,
            "all_clear": len(alerts) == 0,
            "baseline_check": "passed" if len(alerts) == 0 else "failed",
        })

    @tools.register(dangerous=True, category="escalation")
    def escalate_to_oncall(service: str, issue: str, severity: str) -> str:
        """
        Page the on-call engineer with deployment issue details.
        DANGEROUS: wakes someone up. Only use for real issues.
        """
        return json.dumps({
            "status": "paged",
            "on_call": "engineering@company.com",
            "severity": severity,
            "service": service,
            "issue": issue,
            "paged_at": "2026-07-25T14:30:00Z",
            "acknowledged": False,
        })

    @tools.register(category="reporting")
    def generate_deploy_report(services_checked: str, all_healthy: str, issues_found: str) -> str:
        """
        Generate a structured deploy verification report.
        Lists all services checked, their status, and any issues.
        """
        return json.dumps({
            "deploy_id": "deploy-2026-07-25-001",
            "timestamp": "2026-07-25T14:30:00Z",
            "services_checked": services_checked,
            "verdict": "PASS" if all_healthy == "true" else "FAIL",
            "issues": issues_found,
            "next_steps": "Monitor for 30 min" if all_healthy == "true" else "Escalate and consider rollback",
        })

    harness = Harness(
        tools=tools,
        system_prompt=textwrap.dedent("""\
            You are a deploy verification agent. Your job is to:
            1. Check health of all deployed services (api, worker, database, frontend)
            2. Scan error logs from the last 5 minutes
            3. Validate key metrics against baselines
            4. Generate a deploy report
            5. Escalate if anything is unhealthy (requires human approval)

            Be systematic. Check every service. Don't assume things are fine.
            If you see errors or breached metrics, flag them explicitly.
            Only escalate real issues — don't page on-call for warnings.
        """),
        max_steps=8,
    )

    return harness


if __name__ == "__main__":
    print("=" * 60)
    print("Deploy Verification Agent — Example")
    print("=" * 60)

    harness = build_deploy_agent()
    random.seed(42)  # deterministic demo

    task = """
    Verify deployment deploy-2026-07-25-001 just completed.
    Check all 4 services (api, worker, database, frontend):
    - Health endpoint
    - Error logs from last 5 minutes
    - Key metrics vs baseline
    - Generate a deploy report
    - Escalate if anything looks wrong

    Start with the api service, then worker, then database, then frontend.
    """

    print("\nDeploy just finished. Running verification...")
    print()

    run = harness.run_task(task)
    run.print_trace()
