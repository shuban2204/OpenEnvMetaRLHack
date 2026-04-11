"""
Inference Script — Cloud FinOps Environment (v2)
==================================================
Advanced multi-turn agent with pre-computed analysis, task-specific strategies,
conversation memory, and structured reasoning.

MANDATORY ENV VARS:
    API_BASE_URL    The LLM endpoint       (default: https://router.huggingface.co/v1)
    MODEL_NAME      The model identifier   (default: Qwen/Qwen2.5-72B-Instruct)
    HF_TOKEN        Your API key

STDOUT FORMAT:
    [START] task=<task> env=<env> model=<model>
    [STEP]  step=<n> action=<action> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...>
"""

from __future__ import annotations

import json
import math
import os
import re
import textwrap
from typing import Any, Dict, List, Optional

import requests
from openai import OpenAI

# ── Pricing & capacity tables (mirrored from environment for pre-computation) ─

INSTANCE_PRICING = {
    "t3.micro": 7.49, "t3.small": 15.04, "t3.medium": 30.09, "t3.large": 60.12,
    "m5.large": 69.12, "m5.xlarge": 138.24, "m5.2xlarge": 276.48,
    "c5.large": 61.20, "c5.xlarge": 122.40, "c5.2xlarge": 244.80,
    "r5.large": 90.72, "r5.xlarge": 181.44,
}

SPOT_PRICING = {
    "t3.micro": 2.25, "t3.small": 4.51, "t3.medium": 9.03, "t3.large": 18.04,
    "m5.large": 20.74, "m5.xlarge": 41.47, "m5.2xlarge": 82.94,
    "c5.large": 18.36, "c5.xlarge": 36.72, "c5.2xlarge": 73.44,
    "r5.large": 27.22, "r5.xlarge": 54.43,
}

RI_PRICING = {
    "t3.micro": 4.87, "t3.small": 9.78, "t3.medium": 19.56, "t3.large": 39.08,
    "m5.large": 44.93, "m5.xlarge": 89.86, "m5.2xlarge": 179.71,
    "c5.large": 39.78, "c5.xlarge": 79.56, "c5.2xlarge": 159.12,
    "r5.large": 58.97, "r5.xlarge": 117.94,
}

INSTANCE_CAPACITY = {
    "t3.micro": 0.5, "t3.small": 1.0, "t3.medium": 2.0, "t3.large": 4.0,
    "m5.large": 1.0, "m5.xlarge": 2.0, "m5.2xlarge": 4.0,
    "c5.large": 1.0, "c5.xlarge": 2.0, "c5.2xlarge": 4.0,
    "r5.large": 1.0, "r5.xlarge": 2.0,
}

DOWNGRADE_PATHS = {
    "m5.2xlarge": ["m5.xlarge", "m5.large"],
    "m5.xlarge":  ["m5.large"],
    "c5.2xlarge": ["c5.xlarge", "c5.large"],
    "c5.xlarge":  ["c5.large"],
    "r5.xlarge":  ["r5.large"],
    "t3.large":   ["t3.medium", "t3.small", "t3.micro"],
    "t3.medium":  ["t3.small", "t3.micro"],
    "t3.small":   ["t3.micro"],
}

# ── Configuration ─────────────────────────────────────────────────────────────

API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
ENV_URL = os.getenv("ENV_URL", "http://localhost:7860")
BENCHMARK = "cloud_finops_env"
TEMPERATURE = 0.1
MAX_TOKENS = 1024

TASK_IDS = [
    "cleanup_unused_volumes",
    "rightsize_overprovisioned",
    "spot_instance_migration",
    "full_cost_optimization",
    "reserved_instance_planning",
    "comprehensive_fleet_review",
]

TASK_MAX_STEPS = {
    "cleanup_unused_volumes": 10,
    "rightsize_overprovisioned": 12,
    "spot_instance_migration": 15,
    "full_cost_optimization": 20,
    "reserved_instance_planning": 18,
    "comprehensive_fleet_review": 25,
}


# ── Logging (mandatory stdout format) ────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={error or 'null'}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ── Environment HTTP helpers ──────────────────────────────────────────────────

def env_reset(task_id: str) -> Dict[str, Any]:
    resp = requests.post(f"{ENV_URL}/reset", json={"task_id": task_id}, timeout=30)
    resp.raise_for_status()
    return resp.json()


def env_step(action: Dict[str, Any]) -> Dict[str, Any]:
    resp = requests.post(f"{ENV_URL}/step", json={"action": action}, timeout=30)
    resp.raise_for_status()
    return resp.json()


# ═══════════════════════════════════════════════════════════════════════════════
#  PRE-COMPUTED ANALYSIS ENGINE
#  The key insight: LLMs struggle with math but excel at following analysis.
#  We pre-compute all SLA math, eligibility checks, and recommendations,
#  then present them as a decision table the LLM just needs to follow.
# ═══════════════════════════════════════════════════════════════════════════════

class FinOpsAnalyzer:
    """Pre-computes actionable analysis from raw observation data."""

    @staticmethod
    def analyze_volumes(volumes: List[Dict]) -> str:
        """Identify deletable vs. attached volumes."""
        if not volumes:
            return ""
        lines = ["=== VOLUME ANALYSIS ==="]
        deletable = []
        for vol in volumes:
            vid = vol["volume_id"]
            state = vol["state"]
            cost = vol["monthly_cost"]
            attached = vol.get("attached_instance_id") or "NONE"
            if state == "available":
                deletable.append(vid)
                lines.append(f"  {vid}: {vol['size_gb']}GB {vol['volume_type']} | "
                             f"UNATTACHED | ${cost:.2f}/mo → DELETE ✓")
            else:
                lines.append(f"  {vid}: {vol['size_gb']}GB | attached to {attached} | "
                             f"${cost:.2f}/mo → KEEP (in-use)")
        if deletable:
            lines.append(f"\n  ACTION PLAN: Delete {len(deletable)} unattached volumes: "
                         f"{', '.join(deletable)}")
            total_savings = sum(v["monthly_cost"] for v in volumes if v["state"] == "available")
            lines.append(f"  TOTAL SAVINGS: ${total_savings:.2f}/mo")
        else:
            lines.append("  No deletable volumes found.")
        return "\n".join(lines)

    @staticmethod
    def analyze_resize(instances: List[Dict], resize_options: Dict) -> str:
        """Pre-compute SLA math for every valid resize option."""
        lines = ["=== RESIZE ANALYSIS (SLA threshold: effective peak < 80%) ==="]
        candidates = []
        for inst in instances:
            iid = inst["instance_id"]
            itype = inst["instance_type"]
            app = inst["app_name"]
            avg_cpu = inst["cpu_avg_percent"]
            peak_cpu = inst["cpu_peak_percent"]
            cost = inst["monthly_cost"]
            deps = inst.get("dependencies", [])
            cpu_hist = inst.get("cpu_history_7d", [])

            # Use 7d history peak if available (more accurate than reported peak)
            actual_peak = peak_cpu
            if cpu_hist:
                hist_peak = max(cpu_hist)
                actual_peak = max(peak_cpu, hist_peak)

            old_cap = INSTANCE_CAPACITY.get(itype, 1.0)
            downgrades = DOWNGRADE_PATHS.get(itype, [])

            if not downgrades or avg_cpu >= 40:
                continue  # Well-utilized or no downgrade path

            line = f"\n  {iid} ({itype}, {app}): avg={avg_cpu}% peak={actual_peak}% ${cost:.2f}/mo"
            if deps:
                line += f" [DEPS: {', '.join(deps)}]"
            lines.append(line)

            best_option = None
            best_savings = 0
            for target_type in downgrades:
                new_cap = INSTANCE_CAPACITY.get(target_type, 1.0)
                effective_peak = actual_peak * (old_cap / new_cap)
                new_cost = INSTANCE_PRICING.get(target_type, cost)
                savings = cost - new_cost
                safe = effective_peak < 80.0
                marker = "✓ SAFE" if safe else "✗ UNSAFE (SLA violation!)"
                lines.append(
                    f"    → {target_type} (cap={new_cap}x): "
                    f"effective_peak = {actual_peak}% × ({old_cap}/{new_cap}) = "
                    f"{effective_peak:.1f}% [{marker}] saves ${savings:.2f}/mo"
                )
                if safe and savings > best_savings:
                    best_savings = savings
                    best_option = target_type

            if best_option:
                candidates.append((iid, best_option, best_savings))
                lines.append(f"    ★ BEST: resize to {best_option} (saves ${best_savings:.2f}/mo)")
            else:
                lines.append(f"    ⚠ NO safe resize — leave as-is")

        if candidates:
            lines.append(f"\n  ACTION PLAN: Resize {len(candidates)} instances:")
            total = 0
            for iid, target, sav in candidates:
                lines.append(f"    resize_instance({iid}, {target}) → saves ${sav:.2f}/mo")
                total += sav
            lines.append(f"  TOTAL SAVINGS: ${total:.2f}/mo")
        return "\n".join(lines)

    @staticmethod
    def analyze_terminate(instances: List[Dict]) -> str:
        """Identify truly idle instances safe to terminate."""
        lines = ["=== TERMINATION ANALYSIS (idle = avg CPU < 5%) ==="]
        candidates = []
        for inst in instances:
            iid = inst["instance_id"]
            app = inst["app_name"]
            avg_cpu = inst["cpu_avg_percent"]
            peak_cpu = inst["cpu_peak_percent"]
            cost = inst["monthly_cost"]
            deps = inst.get("dependencies", [])
            tags = inst.get("tags", {})
            tier = tags.get("tier", "")

            if avg_cpu >= 5:
                continue

            if deps:
                lines.append(f"  {iid} ({app}): avg={avg_cpu}% ${cost:.2f}/mo "
                             f"→ KEEP (depended on by: {', '.join(deps)}) ✗")
            elif tier == "critical":
                lines.append(f"  {iid} ({app}): avg={avg_cpu}% ${cost:.2f}/mo "
                             f"→ KEEP (critical tier) ✗")
            else:
                candidates.append((iid, cost))
                lines.append(f"  {iid} ({app}): avg={avg_cpu}% peak={peak_cpu}% "
                             f"${cost:.2f}/mo → TERMINATE ✓")

        if candidates:
            total = sum(c for _, c in candidates)
            lines.append(f"\n  ACTION PLAN: Terminate {len(candidates)} idle instances:")
            for iid, cost in candidates:
                lines.append(f"    terminate_instance({iid}) → saves ${cost:.2f}/mo")
            lines.append(f"  TOTAL SAVINGS: ${total:.2f}/mo")
        else:
            lines.append("  No idle instances to terminate.")
        return "\n".join(lines)

    @staticmethod
    def analyze_spot(instances: List[Dict]) -> str:
        """Analyze spot conversion eligibility based on tags and dependencies."""
        lines = ["=== SPOT ELIGIBILITY ANALYSIS ==="]
        lines.append("  Rules: Must be stateless, non-critical, no dependencies")
        candidates = []
        for inst in instances:
            iid = inst["instance_id"]
            itype = inst["instance_type"]
            app = inst["app_name"]
            cost = inst["monthly_cost"]
            deps = inst.get("dependencies", [])
            tags = inst.get("tags", {})
            pricing = inst.get("pricing_model", "on-demand")

            if pricing != "on-demand":
                continue  # Already spot or reserved

            is_stateful = tags.get("stateful", "false") == "true"
            is_critical = tags.get("tier", "") == "critical"
            env_tag = tags.get("env", "")
            has_deps = len(deps) > 0

            reasons_no = []
            if is_stateful:
                reasons_no.append("STATEFUL=true (data loss risk)")
            if is_critical:
                reasons_no.append("CRITICAL tier")
            if has_deps:
                reasons_no.append(f"has dependents: {', '.join(deps)}")

            spot_cost = SPOT_PRICING.get(itype, cost)
            savings = cost - spot_cost

            if reasons_no:
                lines.append(f"  {iid} ({app}, {itype}): ${cost:.2f}/mo "
                             f"→ NOT ELIGIBLE: {'; '.join(reasons_no)} ✗")
            else:
                candidates.append((iid, savings))
                lines.append(f"  {iid} ({app}, {itype}, env={env_tag}): "
                             f"stateful=false, non-critical, no deps "
                             f"→ ELIGIBLE ✓ saves ${savings:.2f}/mo")

        if candidates:
            total = sum(s for _, s in candidates)
            lines.append(f"\n  ACTION PLAN: Convert {len(candidates)} instances to spot:")
            for iid, sav in candidates:
                lines.append(f"    convert_to_spot({iid}) → saves ${sav:.2f}/mo")
            lines.append(f"  TOTAL SAVINGS: ${total:.2f}/mo")
        return "\n".join(lines)

    @staticmethod
    def analyze_ri(instances: List[Dict]) -> str:
        """Analyze RI purchase suitability based on stability and uptime."""
        lines = ["=== RESERVED INSTANCE ANALYSIS ==="]
        lines.append("  Rules: uptime ≥ 180d, CPU std_dev < 10, not deprecated/migrating, prod env")
        candidates = []
        for inst in instances:
            iid = inst["instance_id"]
            itype = inst["instance_type"]
            app = inst["app_name"]
            cost = inst["monthly_cost"]
            tags = inst.get("tags", {})
            uptime = inst.get("uptime_days", 0)
            cpu_hist = inst.get("cpu_history_7d", [])
            pricing = inst.get("pricing_model", "on-demand")
            ri_eligible = inst.get("ri_eligible", False)
            env_tag = tags.get("env", "")

            if pricing != "on-demand":
                continue

            # Compute CPU stability
            std_dev = 0.0
            if cpu_hist and len(cpu_hist) >= 3:
                mean_cpu = sum(cpu_hist) / len(cpu_hist)
                variance = sum((x - mean_cpu) ** 2 for x in cpu_hist) / len(cpu_hist)
                std_dev = math.sqrt(variance)

            status = tags.get("status", "")
            has_decommission = "decommission_date" in tags
            is_deprecated = status in ("deprecated", "migrating")

            reasons_no = []
            if uptime < 180:
                reasons_no.append(f"uptime={uptime}d (<180)")
            if std_dev > 10:
                reasons_no.append(f"CPU std_dev={std_dev:.1f} (>10, too variable)")
            if is_deprecated:
                reasons_no.append(f"status={status}")
            if has_decommission:
                reasons_no.append(f"decommission_date={tags['decommission_date']}")
            if env_tag in ("dev", "staging") and not ri_eligible:
                reasons_no.append(f"env={env_tag} (non-prod)")

            ri_cost = RI_PRICING.get(itype, cost)
            savings = cost - ri_cost

            if reasons_no:
                lines.append(f"  {iid} ({app}, {itype}): uptime={uptime}d "
                             f"std_dev={std_dev:.1f} → NOT SUITABLE: "
                             f"{'; '.join(reasons_no)} ✗")
            else:
                candidates.append((iid, savings))
                lines.append(f"  {iid} ({app}, {itype}): uptime={uptime}d "
                             f"std_dev={std_dev:.1f} env={env_tag} "
                             f"→ GOOD CANDIDATE ✓ saves ${savings:.2f}/mo")

        if candidates:
            total = sum(s for _, s in candidates)
            lines.append(f"\n  ACTION PLAN: Purchase RI for {len(candidates)} instances:")
            for iid, sav in candidates:
                lines.append(f"    purchase_ri({iid}) → saves ${sav:.2f}/mo")
            lines.append(f"  TOTAL SAVINGS: ${total:.2f}/mo")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
#  TASK-SPECIFIC SYSTEM PROMPTS
#  Each task gets an expert prompt tailored to its specific challenge.
# ═══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPTS = {
    "cleanup_unused_volumes": textwrap.dedent("""\
        You are a Cloud Financial Engineer. Your task is simple: delete all
        unattached EBS storage volumes to reduce monthly spend.

        STRATEGY:
        1. Look at the VOLUME ANALYSIS section — it lists which volumes are
           UNATTACHED (state='available') and safe to delete.
        2. Delete each unattached volume one by one using delete_volume.
        3. NEVER delete volumes that are 'in-use' (attached to instances).
        4. After deleting all unattached volumes, use 'submit'.

        Respond with ONLY a JSON action, no explanation:
        {"action_type": "delete_volume", "target_id": "vol-XXX"}
        {"action_type": "submit"}
    """),

    "rightsize_overprovisioned": textwrap.dedent("""\
        You are a Cloud Financial Engineer performing right-sizing.

        CRITICAL: The RESIZE ANALYSIS section has pre-computed the SLA math
        for every valid resize option. It shows the effective peak CPU after
        resizing and whether it's SAFE or UNSAFE.

        STRATEGY:
        1. Read the RESIZE ANALYSIS table carefully.
        2. For each instance marked with ★ BEST, resize to that type.
        3. NEVER resize instances marked ⚠ NO safe resize.
        4. NEVER resize instances where all options show ✗ UNSAFE.
        5. Always use the ★ BEST recommended option (maximum savings while safe).
        6. After all resizes, use 'submit'.

        Respond with ONLY a JSON action, no explanation:
        {"action_type": "resize_instance", "target_id": "i-XXX", "new_type": "m5.large"}
        {"action_type": "submit"}
    """),

    "spot_instance_migration": textwrap.dedent("""\
        You are a Cloud Financial Engineer evaluating spot migration.

        CRITICAL: The SPOT ELIGIBILITY ANALYSIS section has pre-analyzed every
        instance's tags, dependencies, and criticality.

        STRATEGY:
        1. Read the SPOT ELIGIBILITY ANALYSIS table.
        2. Convert ONLY instances marked → ELIGIBLE ✓ to spot.
        3. NEVER convert instances marked → NOT ELIGIBLE ✗.
        4. Key disqualifiers: stateful=true, tier=critical, has dependents.
        5. After converting all eligible instances, use 'submit'.

        Respond with ONLY a JSON action, no explanation:
        {"action_type": "convert_to_spot", "target_id": "i-XXX"}
        {"action_type": "submit"}
    """),

    "full_cost_optimization": textwrap.dedent("""\
        You are a Cloud Financial Engineer doing comprehensive fleet optimization.

        STRATEGY — follow this EXACT priority order:
        1. FIRST: Delete all unattached volumes (see VOLUME ANALYSIS for targets)
        2. SECOND: Terminate idle instances with avg CPU < 5% and NO dependencies
           (see TERMINATION ANALYSIS for safe targets)
        3. THIRD: Resize over-provisioned instances (see RESIZE ANALYSIS —
           ONLY use options marked ✓ SAFE, prefer ★ BEST)
        4. FINALLY: Use 'submit'

        CRITICAL WARNINGS:
        - Check the 7-day CPU history! An instance with low AVERAGE but HIGH
          PEAKS (like weekend batches) will cause SLA violations if resized.
        - NEVER terminate instances that other services depend on.
        - NEVER resize if effective peak would exceed 80%.
        - The analysis sections have pre-computed everything — follow them.

        Respond with ONLY a JSON action, no explanation:
        {"action_type": "delete_volume", "target_id": "vol-XXX"}
        {"action_type": "terminate_instance", "target_id": "i-XXX"}
        {"action_type": "resize_instance", "target_id": "i-XXX", "new_type": "m5.large"}
        {"action_type": "submit"}
    """),

    "reserved_instance_planning": textwrap.dedent("""\
        You are a Cloud Financial Engineer planning Reserved Instance purchases.

        CRITICAL: The RI ANALYSIS section has pre-computed CPU stability
        (std_dev), checked uptime, and verified deprecation status.

        STRATEGY:
        1. Read the RI ANALYSIS table.
        2. Purchase RI ONLY for instances marked → GOOD CANDIDATE ✓.
        3. NEVER purchase RI for instances marked → NOT SUITABLE ✗.
        4. Key disqualifiers: uptime < 180d, CPU std_dev > 10,
           status=deprecated/migrating, decommission_date set.
        5. After purchasing all good RIs, use 'submit'.

        Respond with ONLY a JSON action, no explanation:
        {"action_type": "purchase_ri", "target_id": "i-XXX"}
        {"action_type": "submit"}
    """),

    "comprehensive_fleet_review": textwrap.dedent("""\
        You are a Cloud Financial Engineer doing a full fleet review using
        ALL optimization strategies: volume cleanup, termination, right-sizing,
        spot migration, and RI purchases.

        STRATEGY — follow this EXACT priority order:
        1. Delete all unattached volumes (VOLUME ANALYSIS)
        2. Terminate truly idle instances with no dependencies (TERMINATION ANALYSIS)
        3. Resize over-provisioned instances safely (RESIZE ANALYSIS — ★ BEST only)
        4. Convert eligible stateless workloads to spot (SPOT ELIGIBILITY ANALYSIS)
        5. Purchase RIs for stable long-running instances (RI ANALYSIS)
        6. Use 'submit' when all optimizations are done

        CRITICAL WARNINGS from analysis sections:
        - Follow the ✓/✗ markers in each analysis section exactly.
        - NEVER act on instances/volumes not recommended by the analysis.
        - Each section's ACTION PLAN gives you the exact commands to run.
        - Work through the sections in order (volumes → terminate → resize → spot → RI).

        Respond with ONLY a JSON action, no explanation.
    """),
}

# Default fallback for unknown tasks
DEFAULT_SYSTEM_PROMPT = textwrap.dedent("""\
    You are a Cloud Financial Engineer optimizing cloud infrastructure costs.
    Read the analysis sections carefully and follow their recommendations.
    Respond with ONLY a JSON action — no explanation.
    {"action_type": "skip"} or {"action_type": "submit"} when done.
""")


# ═══════════════════════════════════════════════════════════════════════════════
#  SMART PROMPT BUILDER
#  Constructs observation + pre-computed analysis tailored to each task.
# ═══════════════════════════════════════════════════════════════════════════════

def build_smart_prompt(obs: Dict[str, Any], step: int, task_id: str) -> str:
    """Build a rich user prompt with pre-computed analysis."""
    obs_data = obs.get("observation", obs)
    parts = []

    max_steps = obs_data.get("max_steps", "?")
    parts.append(f"STEP {step}/{max_steps} — {task_id}")
    parts.append(f"Monthly spend: ${obs_data.get('current_monthly_spend', 0):.2f} | "
                 f"Savings so far: ${obs_data.get('savings_achieved', 0):.2f}")

    # Show violations prominently
    violations = obs_data.get("violations", [])
    if violations:
        parts.append(f"\n⚠ VIOLATIONS ({len(violations)}):")
        for v in violations:
            parts.append(f"  - {v}")
        parts.append("  → Avoid similar actions!\n")

    # Show last action error
    last_err = obs_data.get("last_action_error")
    if last_err:
        parts.append(f"⚠ LAST ACTION ERROR: {last_err}")

    # Action history
    history = obs_data.get("action_history", [])
    if history:
        parts.append(f"\nActions taken ({len(history)}): {', '.join(history)}")

    instances = obs_data.get("instances", [])
    volumes = obs_data.get("volumes", [])
    resize_opts = obs_data.get("resize_options", {})

    # ── Task-specific analysis sections ──
    if task_id == "cleanup_unused_volumes":
        parts.append(f"\n{FinOpsAnalyzer.analyze_volumes(volumes)}")

    elif task_id == "rightsize_overprovisioned":
        parts.append(f"\n{FinOpsAnalyzer.analyze_resize(instances, resize_opts)}")

    elif task_id == "spot_instance_migration":
        parts.append(f"\n{FinOpsAnalyzer.analyze_spot(instances)}")

    elif task_id == "reserved_instance_planning":
        parts.append(f"\n{FinOpsAnalyzer.analyze_ri(instances)}")

    elif task_id in ("full_cost_optimization", "comprehensive_fleet_review"):
        # Full optimization needs ALL analyses
        parts.append(f"\n{FinOpsAnalyzer.analyze_volumes(volumes)}")
        parts.append(f"\n{FinOpsAnalyzer.analyze_terminate(instances)}")
        parts.append(f"\n{FinOpsAnalyzer.analyze_resize(instances, resize_opts)}")
        if task_id == "comprehensive_fleet_review":
            parts.append(f"\n{FinOpsAnalyzer.analyze_spot(instances)}")
            parts.append(f"\n{FinOpsAnalyzer.analyze_ri(instances)}")

    # ── Raw instance data (compact, for context) ──
    parts.append(f"\n--- RAW DATA: {len(instances)} instances, {len(volumes)} volumes ---")
    for inst in instances:
        line = (f"  {inst['instance_id']} | {inst['instance_type']} | {inst['app_name']} "
                f"| cpu_avg={inst['cpu_avg_percent']}% peak={inst['cpu_peak_percent']}% "
                f"| ${inst['monthly_cost']:.2f}/mo")
        deps = inst.get("dependencies", [])
        if deps:
            line += f" | deps={deps}"
        tags = inst.get("tags", {})
        if tags:
            line += f" | tags={tags}"
        cpu_hist = inst.get("cpu_history_7d", [])
        if cpu_hist:
            line += f" | 7d_cpu={cpu_hist}"
        uptime = inst.get("uptime_days", 0)
        if uptime:
            line += f" | uptime={uptime}d"
        pricing = inst.get("pricing_model", "")
        if pricing and pricing != "on-demand":
            line += f" | pricing={pricing}"
        parts.append(line)

    parts.append("\nRespond with ONE JSON action:")
    return "\n".join(parts)


# ═══════════════════════════════════════════════════════════════════════════════
#  MULTI-TURN AGENT
#  Maintains conversation history for context across steps.
# ═══════════════════════════════════════════════════════════════════════════════

class FinOpsAgent:
    """Multi-turn LLM agent with conversation memory and pre-computed analysis."""

    MAX_HISTORY_TURNS = 12  # Keep last N turn pairs to avoid context overflow

    def __init__(self, client: OpenAI, model: str, temperature: float = 0.1):
        self.client = client
        self.model = model
        self.temperature = temperature
        self.messages: List[Dict[str, str]] = []
        self.task_id: str = ""

    def reset(self, task_id: str) -> None:
        self.task_id = task_id
        system_prompt = SYSTEM_PROMPTS.get(task_id, DEFAULT_SYSTEM_PROMPT)
        self.messages = [{"role": "system", "content": system_prompt}]

    def act(self, obs: Dict[str, Any], step: int) -> Dict[str, Any]:
        user_prompt = build_smart_prompt(obs, step, self.task_id)
        self.messages.append({"role": "user", "content": user_prompt})

        # Trim history to avoid context overflow (keep system + last N pairs)
        self._trim_history()

        action = self._call_llm()
        return action

    def _trim_history(self) -> None:
        """Keep system message + last MAX_HISTORY_TURNS user/assistant pairs."""
        if len(self.messages) <= 1 + 2 * self.MAX_HISTORY_TURNS:
            return
        system = self.messages[0]
        # Keep the most recent turns
        recent = self.messages[-(2 * self.MAX_HISTORY_TURNS):]
        self.messages = [system] + recent

    def _call_llm(self) -> Dict[str, Any]:
        """Call the LLM with retry logic and robust JSON extraction."""
        for attempt in range(3):
            try:
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=self.messages,
                    temperature=self.temperature,
                    max_tokens=MAX_TOKENS,
                    stream=False,
                )
                text = (completion.choices[0].message.content or "").strip()

                # Record assistant response for conversation memory
                self.messages.append({"role": "assistant", "content": text})

                action = self._extract_action(text)
                if action:
                    return action

                # If extraction failed, ask for clarification on retry
                if attempt < 2:
                    self.messages.append({
                        "role": "user",
                        "content": (
                            "Your response was not valid JSON. Respond with ONLY "
                            "a JSON object like: {\"action_type\": \"submit\"}"
                        ),
                    })

            except Exception as exc:
                print(f"[DEBUG] LLM call attempt {attempt + 1} failed: {exc}", flush=True)

        return {"action_type": "skip"}

    @staticmethod
    def _extract_action(text: str) -> Optional[Dict[str, Any]]:
        """Robustly extract a JSON action from LLM output."""
        # Strip markdown code fences
        cleaned = text
        if "```" in cleaned:
            # Extract content between code fences
            match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", cleaned, re.DOTALL)
            if match:
                cleaned = match.group(1).strip()
            else:
                # Fallback: strip fences manually
                cleaned = cleaned.replace("```json", "").replace("```", "").strip()

        # Try to find JSON object in the text
        # First try: parse the whole thing
        try:
            obj = json.loads(cleaned)
            if isinstance(obj, dict) and "action_type" in obj:
                return obj
        except json.JSONDecodeError:
            pass

        # Second try: find JSON object pattern in text
        json_pattern = re.search(r'\{[^{}]*"action_type"\s*:\s*"[^"]+?"[^{}]*\}', cleaned)
        if json_pattern:
            try:
                obj = json.loads(json_pattern.group())
                if "action_type" in obj:
                    return obj
            except json.JSONDecodeError:
                pass

        # Third try: find any JSON object
        for match in re.finditer(r'\{[^{}]+\}', text):
            try:
                obj = json.loads(match.group())
                if "action_type" in obj:
                    return obj
            except json.JSONDecodeError:
                continue

        return None


# ═══════════════════════════════════════════════════════════════════════════════
#  EPISODE RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

def run_episode(agent: FinOpsAgent, task_id: str) -> None:
    max_steps = TASK_MAX_STEPS.get(task_id, 15)
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_id, env=BENCHMARK, model=agent.model)

    try:
        obs = env_reset(task_id)
        agent.reset(task_id)

        for step in range(1, max_steps + 1):
            if obs.get("done", False):
                break

            action = agent.act(obs, step)
            action_str = action.get("action_type", "skip")
            target = action.get("target_id", "")
            if target:
                action_str += f"({target}"
                nt = action.get("new_type", "")
                if nt:
                    action_str += f",{nt}"
                action_str += ")"

            obs = env_step(action)
            reward = obs.get("reward") or 0.0
            done = obs.get("done", False)

            rewards.append(reward)
            steps_taken = step

            log_step(
                step=step,
                action=action_str,
                reward=reward,
                done=done,
                error=obs.get("last_action_error"),
            )

            if done:
                break

        # Final score
        if rewards:
            score = rewards[-1] if obs.get("done", False) else sum(rewards)
        score = max(0.001, min(score, 0.999))
        success = score >= 0.1

    except Exception as exc:
        print(f"[DEBUG] Episode error: {exc}", flush=True)

    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    agent = FinOpsAgent(client, MODEL_NAME, temperature=TEMPERATURE)

    for task_id in TASK_IDS:
        print(f"\n{'=' * 60}", flush=True)
        print(f"  Running task: {task_id}", flush=True)
        print(f"{'=' * 60}\n", flush=True)
        run_episode(agent, task_id)


if __name__ == "__main__":
    main()
