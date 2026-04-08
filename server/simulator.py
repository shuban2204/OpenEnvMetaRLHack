"""
Cloud Infrastructure Simulator
================================
Generates deterministic, realistic cloud infrastructure data for each task.
All data is seeded so grading is reproducible.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ── Pricing tables (duplicated here to keep server self-contained) ──────────

INSTANCE_PRICING: Dict[str, float] = {
    "t3.micro":    7.49,
    "t3.small":   15.04,
    "t3.medium":  30.09,
    "t3.large":   60.12,
    "m5.large":   69.12,
    "m5.xlarge": 138.24,
    "m5.2xlarge":276.48,
    "c5.large":   61.20,
    "c5.xlarge": 122.40,
    "c5.2xlarge":244.80,
    "r5.large":   90.72,
    "r5.xlarge": 181.44,
}

INSTANCE_CAPACITY: Dict[str, float] = {
    "t3.micro":   0.5, "t3.small":  1.0, "t3.medium": 2.0, "t3.large":  4.0,
    "m5.large":   1.0, "m5.xlarge": 2.0, "m5.2xlarge":4.0,
    "c5.large":   1.0, "c5.xlarge": 2.0, "c5.2xlarge":4.0,
    "r5.large":   1.0, "r5.xlarge": 2.0,
}

VOLUME_PRICING: Dict[str, float] = {"gp3": 0.08, "gp2": 0.10, "io1": 0.125}

DOWNGRADE_PATHS: Dict[str, List[str]] = {
    "m5.2xlarge": ["m5.xlarge", "m5.large"],
    "m5.xlarge":  ["m5.large"],
    "c5.2xlarge": ["c5.xlarge", "c5.large"],
    "c5.xlarge":  ["c5.large"],
    "r5.xlarge":  ["r5.large"],
    "t3.large":   ["t3.medium", "t3.small", "t3.micro"],
    "t3.medium":  ["t3.small", "t3.micro"],
    "t3.small":   ["t3.micro"],
}


# ── Data classes ────────────────────────────────────────────────────────────

@dataclass
class Instance:
    instance_id: str
    instance_type: str
    state: str                     # running | stopped | terminated
    app_name: str
    cpu_avg_percent: float
    cpu_peak_percent: float
    memory_avg_percent: float
    monthly_cost: float
    cpu_history_7d: List[float] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)  # apps that depend on THIS
    # grading helpers (not exposed to agent)
    optimal_action: str = "keep"   # keep | terminate | resize
    optimal_type: str = ""         # target type for resize

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "instance_id": self.instance_id,
            "instance_type": self.instance_type,
            "state": self.state,
            "app_name": self.app_name,
            "cpu_avg_percent": self.cpu_avg_percent,
            "cpu_peak_percent": self.cpu_peak_percent,
            "memory_avg_percent": self.memory_avg_percent,
            "monthly_cost": self.monthly_cost,
        }
        if self.cpu_history_7d:
            d["cpu_history_7d"] = self.cpu_history_7d
        if self.dependencies:
            d["dependencies"] = self.dependencies
        return d


@dataclass
class Volume:
    volume_id: str
    size_gb: int
    volume_type: str               # gp3 | gp2 | io1
    state: str                     # in-use | available
    attached_instance_id: Optional[str] = None
    monthly_cost: float = 0.0
    # grading helper
    optimal_action: str = "keep"   # keep | delete

    def __post_init__(self) -> None:
        if self.monthly_cost == 0.0:
            self.monthly_cost = round(
                self.size_gb * VOLUME_PRICING.get(self.volume_type, 0.08), 2
            )

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "volume_id": self.volume_id,
            "size_gb": self.size_gb,
            "volume_type": self.volume_type,
            "state": self.state,
            "attached_instance_id": self.attached_instance_id,
            "monthly_cost": self.monthly_cost,
        }
        return d


@dataclass
class CloudState:
    """Mutable snapshot of the simulated cloud infrastructure."""
    instances: List[Instance]
    volumes: List[Volume]

    def total_monthly_spend(self) -> float:
        inst_cost = sum(
            i.monthly_cost for i in self.instances if i.state == "running"
        )
        vol_cost = sum(
            v.monthly_cost for v in self.volumes if v.state != "deleted"
        )
        return round(inst_cost + vol_cost, 2)

    def deep_copy(self) -> "CloudState":
        return CloudState(
            instances=[copy.deepcopy(i) for i in self.instances],
            volumes=[copy.deepcopy(v) for v in self.volumes],
        )


# ── Task-specific infrastructure generators ─────────────────────────────────

def _build_easy_infra() -> CloudState:
    """Task 1: Cleanup unused volumes — 6 instances, 10 volumes (4 unattached)."""
    instances = [
        Instance("i-001", "t3.medium",  "running", "web-frontend",  45, 72, 40, 30.09),
        Instance("i-002", "m5.large",   "running", "api-server",    62, 85, 55, 69.12),
        Instance("i-003", "t3.small",   "running", "monitoring",    28, 41, 30, 15.04),
        Instance("i-004", "c5.large",   "running", "batch-worker",  55, 78, 42, 61.20),
        Instance("i-005", "t3.micro",   "running", "bastion",       12, 25, 15,  7.49),
        Instance("i-006", "m5.xlarge",  "running", "database",      70, 92, 68, 138.24),
    ]
    volumes = [
        Volume("vol-001", 100, "gp3", "in-use",    "i-001"),  # keep
        Volume("vol-002", 500, "gp3", "in-use",    "i-002"),  # keep
        Volume("vol-003",  50, "gp3", "available",  None, optimal_action="delete"),
        Volume("vol-004", 200, "gp2", "in-use",    "i-003"),  # keep
        Volume("vol-005", 100, "gp3", "in-use",    "i-004"),  # keep
        Volume("vol-006", 250, "io1", "available",  None, optimal_action="delete"),
        Volume("vol-007",  50, "gp3", "in-use",    "i-005"),  # keep
        Volume("vol-008",1000, "gp3", "available",  None, optimal_action="delete"),
        Volume("vol-009", 500, "gp2", "in-use",    "i-006"),  # keep
        Volume("vol-010",  75, "gp2", "available",  None, optimal_action="delete"),
    ]
    return CloudState(instances, volumes)


def _build_medium_infra() -> CloudState:
    """Task 2: Right-size over-provisioned instances — 8 instances, 4 volumes."""
    instances = [
        Instance("i-101", "m5.2xlarge", "running", "legacy-api",    8, 15, 12, 276.48,
                 optimal_action="resize", optimal_type="m5.large"),
        Instance("i-102", "c5.2xlarge", "running", "test-runner",   5, 10,  8, 244.80,
                 optimal_action="resize", optimal_type="c5.large"),
        Instance("i-103", "r5.xlarge",  "running", "cache",        65, 88, 72, 181.44),
        Instance("i-104", "m5.xlarge",  "running", "ml-pipeline",  12, 22, 18, 138.24,
                 optimal_action="resize", optimal_type="m5.large"),
        Instance("i-105", "t3.large",   "running", "staging-web",   7, 14, 10,  60.12,
                 optimal_action="resize", optimal_type="t3.small"),
        Instance("i-106", "c5.xlarge",  "running", "data-proc",    58, 80, 45, 122.40),
        Instance("i-107", "m5.xlarge",  "running", "analytics",     9, 18, 14, 138.24,
                 optimal_action="resize", optimal_type="m5.large"),
        Instance("i-108", "t3.medium",  "running", "log-collector", 42, 65, 38,  30.09),
    ]
    volumes = [
        Volume("vol-101", 200, "gp3", "in-use", "i-101"),
        Volume("vol-102", 500, "gp3", "in-use", "i-103"),
        Volume("vol-103", 100, "gp3", "in-use", "i-106"),
        Volume("vol-104", 300, "gp3", "in-use", "i-108"),
    ]
    return CloudState(instances, volumes)


def _build_hard_infra() -> CloudState:
    """Task 3: Full cost optimization — 15 instances, 12 volumes, dependencies."""
    instances = [
        Instance("i-201", "m5.2xlarge", "running", "prod-api",      55, 82, 60, 276.48,
                 cpu_history_7d=[52, 54, 55, 58, 55, 53, 56],
                 dependencies=["staging-api"]),
        Instance("i-202", "m5.xlarge",  "running", "staging-api",    3,  8,  5, 138.24,
                 cpu_history_7d=[3, 2, 4, 3, 3, 2, 4],
                 optimal_action="terminate"),
        Instance("i-203", "c5.2xlarge", "running", "ml-training",    6, 12,  9, 244.80,
                 cpu_history_7d=[5, 6, 8, 5, 7, 6, 5],
                 optimal_action="resize", optimal_type="c5.large"),
        Instance("i-204", "t3.large",   "running", "dev-server",     2,  5,  3,  60.12,
                 cpu_history_7d=[2, 1, 3, 2, 2, 1, 2],
                 optimal_action="terminate"),
        Instance("i-205", "r5.xlarge",  "running", "prod-db",       72, 95, 80, 181.44,
                 cpu_history_7d=[70, 72, 75, 72, 68, 73, 74],
                 dependencies=["prod-api", "analytics", "etl-pipeline"]),
        Instance("i-206", "m5.xlarge",  "running", "analytics",     11, 20, 15, 138.24,
                 cpu_history_7d=[10, 12, 11, 13, 10, 9, 12],
                 optimal_action="resize", optimal_type="m5.large"),
        Instance("i-207", "c5.xlarge",  "running", "batch-jobs",    45, 70, 40, 122.40,
                 cpu_history_7d=[42, 45, 48, 50, 43, 40, 47]),
        Instance("i-208", "t3.medium",  "running", "monitoring",    35, 55, 30,  30.09,
                 cpu_history_7d=[33, 35, 36, 38, 34, 32, 35],
                 dependencies=["prod-api", "batch-jobs", "etl-pipeline"]),
        Instance("i-209", "m5.2xlarge", "running", "old-reports",    1,  3,  2, 276.48,
                 cpu_history_7d=[1, 1, 2, 1, 1, 0, 1],
                 optimal_action="terminate"),
        Instance("i-210", "t3.large",   "running", "qa-server",      4,  9,  6,  60.12,
                 cpu_history_7d=[4, 3, 5, 4, 4, 3, 5],
                 optimal_action="terminate"),
        Instance("i-211", "c5.xlarge",  "running", "etl-pipeline",  15, 28, 20, 122.40,
                 cpu_history_7d=[14, 15, 16, 18, 14, 13, 16],
                 optimal_action="resize", optimal_type="c5.large"),
        Instance("i-212", "m5.xlarge",  "running", "search-svc",    52, 78, 55, 138.24,
                 cpu_history_7d=[50, 52, 55, 53, 51, 49, 54],
                 dependencies=["prod-api", "batch-jobs"]),
        Instance("i-213", "r5.large",   "running", "cache-layer",   60, 85, 65,  90.72,
                 cpu_history_7d=[58, 60, 62, 63, 59, 57, 61],
                 dependencies=["prod-api"]),
        Instance("i-214", "t3.small",   "running", "cron-jobs",      8, 18, 10,  15.04,
                 cpu_history_7d=[7, 8, 9, 8, 7, 6, 9]),
        Instance("i-215", "m5.xlarge",  "running", "log-agg",       10, 19, 13, 138.24,
                 cpu_history_7d=[9, 10, 11, 12, 10, 9, 11],
                 optimal_action="resize", optimal_type="m5.large"),
        # TRAP: Low avg CPU but massive weekend spike (day 6-7).
        # Avg=18% looks like a resize candidate, but peak=72% on weekends
        # means m5.large (1x) would hit effective peak 72*2=144% → SLA violation!
        # Agent must read 7d history to avoid this trap. Correct action: KEEP.
        Instance("i-216", "m5.xlarge",  "running", "weekend-batch", 18, 72, 22, 138.24,
                 cpu_history_7d=[5, 6, 4, 5, 7, 68, 72]),
    ]
    volumes = [
        Volume("vol-201", 500, "gp3", "in-use",    "i-201"),
        Volume("vol-202", 200, "gp3", "available",  None, optimal_action="delete"),
        Volume("vol-203",1000, "io1", "in-use",    "i-205"),
        Volume("vol-204", 100, "gp3", "available",  None, optimal_action="delete"),
        Volume("vol-205", 300, "gp2", "in-use",    "i-207"),
        Volume("vol-206",  50, "gp3", "in-use",    "i-208"),
        Volume("vol-207", 500, "gp3", "available",  None, optimal_action="delete"),
        Volume("vol-208", 200, "gp3", "in-use",    "i-212"),
        Volume("vol-209", 100, "gp2", "available",  None, optimal_action="delete"),
        Volume("vol-210", 750, "gp3", "in-use",    "i-213"),
        Volume("vol-211", 100, "gp3", "in-use",    "i-201"),
        Volume("vol-212", 150, "gp3", "available",  None, optimal_action="delete"),
    ]
    return CloudState(instances, volumes)


# ── Public API ──────────────────────────────────────────────────────────────

TASK_BUILDERS = {
    "cleanup_unused_volumes":   _build_easy_infra,
    "rightsize_overprovisioned": _build_medium_infra,
    "full_cost_optimization":   _build_hard_infra,
}


def build_infrastructure(task_id: str) -> CloudState:
    builder = TASK_BUILDERS.get(task_id)
    if builder is None:
        raise ValueError(
            f"Unknown task_id={task_id!r}. "
            f"Choose from: {list(TASK_BUILDERS.keys())}"
        )
    return builder()
