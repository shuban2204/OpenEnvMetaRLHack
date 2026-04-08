"""
Task Definitions & Graders
============================
Five tasks (easy -> medium -> medium-hard -> hard -> expert) with
deterministic programmatic graders. Each grader produces a score in [0.0, 1.0].
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from server.simulator import (
    DOWNGRADE_PATHS,
    INSTANCE_CAPACITY,
    INSTANCE_PRICING,
    RI_PRICING,
    SPOT_PRICING,
    CloudState,
    build_infrastructure,
)


# ── Task metadata ───────────────────────────────────────────────────────────

@dataclass
class TaskDef:
    task_id: str
    name: str
    difficulty: str
    max_steps: int
    description: str


TASKS: Dict[str, TaskDef] = {
    "cleanup_unused_volumes": TaskDef(
        task_id="cleanup_unused_volumes",
        name="Cleanup Unused Volumes",
        difficulty="easy",
        max_steps=10,
        description=(
            "You are a Cloud Financial Engineer reviewing an AWS account. "
            "Your goal is to REDUCE monthly cloud spend by deleting all "
            "unattached (state='available') EBS storage volumes. "
            "Do NOT delete volumes that are attached to running instances "
            "(state='in-use'). Use 'delete_volume' for each unattached volume, "
            "then 'submit' when done."
        ),
    ),
    "rightsize_overprovisioned": TaskDef(
        task_id="rightsize_overprovisioned",
        name="Right-Size Over-Provisioned Instances",
        difficulty="medium",
        max_steps=12,
        description=(
            "You are a Cloud Financial Engineer performing a right-sizing audit. "
            "Several EC2 instances are heavily over-provisioned (very low CPU "
            "utilization). Your goal is to resize them to smaller, cheaper "
            "instance types WITHOUT causing SLA violations. "
            "An SLA violation occurs if the instance's peak CPU load would "
            "exceed 80% of the new type's capacity. "
            "Within a family, capacity halves with each step down "
            "(e.g. m5.2xlarge=4x, m5.xlarge=2x, m5.large=1x). "
            "Resize under-utilized instances (avg CPU < 25%) to the smallest "
            "safe type. Leave well-utilized instances (avg CPU >= 40%) alone. "
            "Use 'resize_instance' with the target 'new_type', then 'submit'."
        ),
    ),
    "spot_instance_migration": TaskDef(
        task_id="spot_instance_migration",
        name="Spot Instance Migration",
        difficulty="medium-hard",
        max_steps=15,
        description=(
            "You are a Cloud Financial Engineer evaluating which instances can "
            "be safely migrated to spot pricing (60-70% savings). "
            "Use 'convert_to_spot' for eligible instances.\n\n"
            "SAFE for spot (convert these):\n"
            "- Stateless workloads (tag stateful='false')\n"
            "- Non-critical: dev, staging, qa environments\n"
            "- Batch/CI workers that can tolerate interruption\n"
            "- No dependencies on this instance from other services\n\n"
            "NOT safe for spot (leave these alone):\n"
            "- Stateful instances (tag stateful='true') — e.g. databases, session stores\n"
            "- Critical production services (tag tier='critical')\n"
            "- Instances with dependencies (other services depend on them)\n\n"
            "Check BOTH tags AND dependencies carefully. "
            "Use 'submit' when done."
        ),
    ),
    "full_cost_optimization": TaskDef(
        task_id="full_cost_optimization",
        name="Full Cost Optimization",
        difficulty="hard",
        max_steps=20,
        description=(
            "You are a Cloud Financial Engineer performing a comprehensive cost "
            "optimization across an entire AWS fleet. You must:\n"
            "1) DELETE all unattached storage volumes (state='available').\n"
            "2) TERMINATE truly idle instances (avg CPU < 5%) that have NO "
            "   other services depending on them (check 'dependencies' list — "
            "   if other apps depend on this instance, do NOT terminate it).\n"
            "3) RESIZE over-provisioned instances (avg CPU < 25%, not idle) "
            "   to the smallest safe type (peak CPU must stay below 80% of "
            "   new capacity). Use the 7-day CPU history to gauge peak load.\n"
            "4) Leave well-utilized and critical instances untouched.\n"
            "Watch out for instances with low AVERAGE but high PEAK CPU in "
            "their 7-day history — resizing them will cause SLA violations.\n"
            "Maximize dollar savings while keeping all production apps running. "
            "Use 'submit' when done."
        ),
    ),
    "reserved_instance_planning": TaskDef(
        task_id="reserved_instance_planning",
        name="Reserved Instance Planning",
        difficulty="expert",
        max_steps=18,
        description=(
            "You are a Cloud Financial Engineer planning Reserved Instance (RI) "
            "purchases. RIs provide 30-40% savings over on-demand but require "
            "a 1-year commitment. Use 'purchase_ri' for good candidates.\n\n"
            "GOOD RI candidates (purchase these):\n"
            "- Long-running instances (uptime_days >= 180)\n"
            "- Stable, predictable CPU usage (low variance in 7-day history)\n"
            "- Production environment workloads that won't be decommissioned\n"
            "- ri_eligible=true in the observation\n\n"
            "BAD RI candidates (do NOT purchase):\n"
            "- Dev/staging/temporary instances\n"
            "- Instances with highly variable CPU (spiky 7-day history)\n"
            "- Instances tagged status='deprecated' or with decommission_date\n"
            "- Recently launched instances (uptime < 180 days)\n"
            "- Instances tagged status='migrating'\n\n"
            "Analyze 7-day CPU variance: std_dev > 10 means too variable. "
            "Check tags carefully for deprecation/migration signals. "
            "Use 'submit' when done."
        ),
    ),
}

TASK_IDS: List[str] = list(TASKS.keys())


# ── Grading helpers ─────────────────────────────────────────────────────────

def _compute_optimal_savings(task_id: str, initial: CloudState) -> float:
    """Sum of all possible savings if every optimal action is taken."""
    savings = 0.0
    for inst in initial.instances:
        if inst.optimal_action == "terminate":
            savings += inst.monthly_cost
        elif inst.optimal_action == "resize" and inst.optimal_type:
            new_cost = INSTANCE_PRICING.get(inst.optimal_type, inst.monthly_cost)
            savings += inst.monthly_cost - new_cost
        elif inst.optimal_action == "convert_to_spot":
            spot_cost = SPOT_PRICING.get(inst.instance_type, inst.monthly_cost)
            savings += inst.monthly_cost - spot_cost
        elif inst.optimal_action == "purchase_ri":
            ri_cost = RI_PRICING.get(inst.instance_type, inst.monthly_cost)
            savings += inst.monthly_cost - ri_cost
    for vol in initial.volumes:
        if vol.optimal_action == "delete":
            savings += vol.monthly_cost
    return round(savings, 2)


def _check_resize_violation(
    old_type: str, new_type: str, cpu_peak: float
) -> bool:
    """Return True if resizing would cause an SLA violation."""
    old_cap = INSTANCE_CAPACITY.get(old_type, 1.0)
    new_cap = INSTANCE_CAPACITY.get(new_type, 1.0)
    if new_cap >= old_cap:
        return False
    effective_peak = cpu_peak * (old_cap / new_cap)
    return effective_peak > 80.0


# ── Per-step reward calculator ──────────────────────────────────────────────

def compute_step_reward(
    task_id: str,
    action_type: str,
    target_id: str,
    new_type: str,
    current_state: CloudState,
    initial_state: CloudState,
) -> tuple[float, float, str | None]:
    """
    Returns (reward, savings_delta, violation_message | None).
    """
    optimal_savings = _compute_optimal_savings(task_id, initial_state)
    if optimal_savings <= 0:
        optimal_savings = 1.0

    if action_type == "skip":
        return (-0.02, 0.0, None)

    if action_type == "submit":
        return (0.0, 0.0, None)

    # ── delete_volume ───────────────────────────────────────────────────
    if action_type == "delete_volume":
        vol = next((v for v in current_state.volumes if v.volume_id == target_id), None)
        if vol is None:
            return (-0.05, 0.0, f"Volume {target_id} not found")
        if vol.state == "deleted":
            return (-0.05, 0.0, f"Volume {target_id} already deleted")
        if vol.state == "in-use":
            return (-0.25, 0.0, f"VIOLATION: Deleted attached volume {target_id} — potential data loss!")
        savings = vol.monthly_cost
        reward = savings / optimal_savings
        return (round(reward, 4), savings, None)

    # ── terminate_instance ──────────────────────────────────────────────
    if action_type == "terminate_instance":
        inst = next((i for i in current_state.instances if i.instance_id == target_id), None)
        if inst is None:
            return (-0.05, 0.0, f"Instance {target_id} not found")
        if inst.state == "terminated":
            return (-0.05, 0.0, f"Instance {target_id} already terminated")
        if inst.dependencies:
            return (
                -0.30, 0.0,
                f"VIOLATION: Terminated {target_id} ({inst.app_name}) which is "
                f"depended on by: {', '.join(inst.dependencies)}",
            )
        ref = next((i for i in initial_state.instances if i.instance_id == target_id), None)
        if ref and ref.optimal_action == "terminate":
            savings = inst.monthly_cost
            reward = savings / optimal_savings
            return (round(reward, 4), savings, None)
        return (
            -0.20, 0.0,
            f"VIOLATION: Terminated active instance {target_id} ({inst.app_name}, "
            f"CPU avg={inst.cpu_avg_percent}%)",
        )

    # ── resize_instance ─────────────────────────────────────────────────
    if action_type == "resize_instance":
        inst = next((i for i in current_state.instances if i.instance_id == target_id), None)
        if inst is None:
            return (-0.05, 0.0, f"Instance {target_id} not found")
        if inst.state == "terminated":
            return (-0.05, 0.0, f"Instance {target_id} is terminated")
        allowed = DOWNGRADE_PATHS.get(inst.instance_type, [])
        if new_type not in allowed and new_type not in INSTANCE_PRICING:
            return (-0.05, 0.0, f"Invalid target type {new_type} for {inst.instance_type}")
        if new_type not in allowed:
            return (-0.05, 0.0, f"Cannot resize {inst.instance_type} to {new_type} (not a valid downgrade)")
        if _check_resize_violation(inst.instance_type, new_type, inst.cpu_peak_percent):
            return (
                -0.20, 0.0,
                f"VIOLATION: Resizing {target_id} to {new_type} would exceed 80% "
                f"capacity (peak CPU {inst.cpu_peak_percent}%)",
            )
        old_cost = inst.monthly_cost
        new_cost = INSTANCE_PRICING[new_type]
        savings = old_cost - new_cost
        if savings <= 0:
            return (-0.02, 0.0, None)
        ref = next((i for i in initial_state.instances if i.instance_id == target_id), None)
        if ref and ref.optimal_action == "resize" and new_type == ref.optimal_type:
            reward = savings / optimal_savings
        else:
            reward = (savings / optimal_savings) * 0.5
        return (round(reward, 4), round(savings, 2), None)

    # ── convert_to_spot ─────────────────────────────────────────────────
    if action_type == "convert_to_spot":
        inst = next((i for i in current_state.instances if i.instance_id == target_id), None)
        if inst is None:
            return (-0.05, 0.0, f"Instance {target_id} not found")
        if inst.state == "terminated":
            return (-0.05, 0.0, f"Instance {target_id} is terminated")
        if inst.pricing_model == "spot":
            return (-0.05, 0.0, f"Instance {target_id} already on spot pricing")
        ref = next((i for i in initial_state.instances if i.instance_id == target_id), None)
        # Check if this is a safe spot conversion
        if inst.dependencies:
            return (
                -0.25, 0.0,
                f"VIOLATION: Converted {target_id} ({inst.app_name}) to spot but it has "
                f"dependents: {', '.join(inst.dependencies)} — service may go down on interruption",
            )
        tags = inst.tags if hasattr(inst, 'tags') else {}
        is_stateful = tags.get("stateful", "false") == "true"
        is_critical = tags.get("tier", "") == "critical"
        if is_stateful:
            return (
                -0.25, 0.0,
                f"VIOLATION: Converted stateful instance {target_id} ({inst.app_name}) "
                f"to spot — data loss risk on interruption",
            )
        if is_critical:
            return (
                -0.20, 0.0,
                f"VIOLATION: Converted critical instance {target_id} ({inst.app_name}) "
                f"to spot — availability risk",
            )
        spot_cost = SPOT_PRICING.get(inst.instance_type, inst.monthly_cost)
        savings = inst.monthly_cost - spot_cost
        if ref and ref.optimal_action == "convert_to_spot":
            reward = savings / optimal_savings
        else:
            reward = (savings / optimal_savings) * 0.5
        return (round(reward, 4), round(savings, 2), None)

    # ── purchase_ri ─────────────────────────────────────────────────────
    if action_type == "purchase_ri":
        inst = next((i for i in current_state.instances if i.instance_id == target_id), None)
        if inst is None:
            return (-0.05, 0.0, f"Instance {target_id} not found")
        if inst.state == "terminated":
            return (-0.05, 0.0, f"Instance {target_id} is terminated")
        if inst.pricing_model == "reserved":
            return (-0.05, 0.0, f"Instance {target_id} already has RI")
        ref = next((i for i in initial_state.instances if i.instance_id == target_id), None)
        tags = inst.tags if hasattr(inst, 'tags') else {}
        is_deprecated = tags.get("status", "") in ("deprecated", "migrating")
        has_decommission = "decommission_date" in tags
        if is_deprecated or has_decommission:
            return (
                -0.25, 0.0,
                f"VIOLATION: Purchased RI for {target_id} ({inst.app_name}) which is "
                f"marked for decommission — wasted 1-year commitment",
            )
        # Check CPU variance from 7d history
        cpu_hist = inst.cpu_history_7d
        if cpu_hist and len(cpu_hist) >= 3:
            mean_cpu = sum(cpu_hist) / len(cpu_hist)
            variance = sum((x - mean_cpu) ** 2 for x in cpu_hist) / len(cpu_hist)
            std_dev = variance ** 0.5
            if std_dev > 15:
                return (
                    -0.15, 0.0,
                    f"VIOLATION: Purchased RI for {target_id} ({inst.app_name}) with "
                    f"highly variable CPU (std_dev={std_dev:.1f}) — usage unpredictable",
                )
        if inst.uptime_days < 180:
            return (
                -0.10, 0.0,
                f"VIOLATION: Purchased RI for {target_id} ({inst.app_name}) with only "
                f"{inst.uptime_days} days uptime — too new for 1-year commitment",
            )
        ri_cost = RI_PRICING.get(inst.instance_type, inst.monthly_cost)
        savings = inst.monthly_cost - ri_cost
        if ref and ref.optimal_action == "purchase_ri":
            reward = savings / optimal_savings
        else:
            reward = (savings / optimal_savings) * 0.3
        return (round(reward, 4), round(savings, 2), None)

    return (-0.05, 0.0, f"Unknown action_type: {action_type}")


# ── Final episode grader ────────────────────────────────────────────────────

def grade_episode(
    task_id: str,
    savings_achieved: float,
    violations: List[str],
    initial_state: CloudState,
    current_state: CloudState,
) -> float:
    """Compute a final score in [0.0, 1.0] for the completed episode."""
    optimal = _compute_optimal_savings(task_id, initial_state)
    if optimal <= 0:
        return 1.0

    savings_ratio = min(savings_achieved / optimal, 1.0)

    if task_id == "cleanup_unused_volumes":
        penalty = 0.25 * len(violations)
        return round(max(savings_ratio - penalty, 0.0), 4)

    if task_id == "rightsize_overprovisioned":
        penalty = 0.20 * len(violations)
        return round(max(savings_ratio - penalty, 0.0), 4)

    if task_id == "spot_instance_migration":
        # 60% savings, 40% zero-violations
        violation_score = 1.0 if len(violations) == 0 else max(0.0, 1.0 - 0.20 * len(violations))
        score = 0.60 * savings_ratio + 0.40 * violation_score
        return round(max(min(score, 1.0), 0.0), 4)

    if task_id == "full_cost_optimization":
        # Hard: 50% savings, 25% zero-violations, 25% completeness
        violation_score = 1.0 if len(violations) == 0 else max(0.0, 1.0 - 0.15 * len(violations))
        total_optimal = 0
        completed_optimal = 0
        for inst in initial_state.instances:
            if inst.optimal_action != "keep":
                total_optimal += 1
                cur = next((c for c in current_state.instances if c.instance_id == inst.instance_id), None)
                if cur is None:
                    continue
                if inst.optimal_action == "terminate" and cur.state == "terminated":
                    completed_optimal += 1
                elif inst.optimal_action == "resize" and cur.instance_type == inst.optimal_type:
                    completed_optimal += 1
        for vol in initial_state.volumes:
            if vol.optimal_action == "delete":
                total_optimal += 1
                cur = next((c for c in current_state.volumes if c.volume_id == vol.volume_id), None)
                if cur and cur.state == "deleted":
                    completed_optimal += 1
        completeness = completed_optimal / total_optimal if total_optimal > 0 else 1.0
        score = 0.50 * savings_ratio + 0.25 * violation_score + 0.25 * completeness
        return round(max(min(score, 1.0), 0.0), 4)

    if task_id == "reserved_instance_planning":
        # Expert: 40% savings, 30% zero-violations, 30% precision
        violation_score = 1.0 if len(violations) == 0 else max(0.0, 1.0 - 0.20 * len(violations))
        # Precision: of all purchase_ri actions, how many were correct?
        total_ri_actions = 0
        correct_ri_actions = 0
        for inst in initial_state.instances:
            cur = next((c for c in current_state.instances if c.instance_id == inst.instance_id), None)
            if cur and cur.pricing_model == "reserved":
                total_ri_actions += 1
                if inst.optimal_action == "purchase_ri":
                    correct_ri_actions += 1
        precision = correct_ri_actions / total_ri_actions if total_ri_actions > 0 else 1.0
        score = 0.40 * savings_ratio + 0.30 * violation_score + 0.30 * precision
        return round(max(min(score, 1.0), 0.0), 4)

    return round(max(savings_ratio, 0.0), 4)
