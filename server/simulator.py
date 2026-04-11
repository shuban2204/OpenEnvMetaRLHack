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


# ── Pricing tables ──────────────────────────────────────────────────────────

INSTANCE_PRICING: Dict[str, float] = {
    "t3.micro":    7.49, "t3.small":  15.04, "t3.medium": 30.09,
    "t3.large":   60.12,
    "m5.large":   69.12, "m5.xlarge":138.24, "m5.2xlarge":276.48,
    "c5.large":   61.20, "c5.xlarge":122.40, "c5.2xlarge":244.80,
    "r5.large":   90.72, "r5.xlarge":181.44,
}

SPOT_PRICING: Dict[str, float] = {
    "t3.micro":    2.25, "t3.small":   4.51, "t3.medium":  9.03,
    "t3.large":   18.04,
    "m5.large":   20.74, "m5.xlarge": 41.47, "m5.2xlarge": 82.94,
    "c5.large":   18.36, "c5.xlarge": 36.72, "c5.2xlarge": 73.44,
    "r5.large":   27.22, "r5.xlarge": 54.43,
}

RI_PRICING: Dict[str, float] = {
    "t3.micro":    4.87, "t3.small":   9.78, "t3.medium": 19.56,
    "t3.large":   39.08,
    "m5.large":   44.93, "m5.xlarge": 89.86, "m5.2xlarge":179.71,
    "c5.large":   39.78, "c5.xlarge": 79.56, "c5.2xlarge":159.12,
    "r5.large":   58.97, "r5.xlarge":117.94,
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
    state: str
    app_name: str
    cpu_avg_percent: float
    cpu_peak_percent: float
    memory_avg_percent: float
    monthly_cost: float
    cpu_history_7d: List[float] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    # Enriched metadata
    tags: Dict[str, str] = field(default_factory=dict)
    region: str = "us-east-1"
    launch_date: str = "2025-06-15"
    pricing_model: str = "on-demand"
    spot_eligible: bool = False
    ri_eligible: bool = False
    network_io_gbps: float = 0.0
    uptime_days: int = 180
    # Grading helpers (not exposed to agent)
    optimal_action: str = "keep"
    optimal_type: str = ""

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
            "tags": self.tags,
            "region": self.region,
            "launch_date": self.launch_date,
            "pricing_model": self.pricing_model,
            "network_io_gbps": self.network_io_gbps,
            "uptime_days": self.uptime_days,
        }
        if self.cpu_history_7d:
            d["cpu_history_7d"] = self.cpu_history_7d
        if self.dependencies:
            d["dependencies"] = self.dependencies
        if self.spot_eligible:
            d["spot_eligible"] = True
        if self.ri_eligible:
            d["ri_eligible"] = True
        return d


@dataclass
class Volume:
    volume_id: str
    size_gb: int
    volume_type: str
    state: str
    attached_instance_id: Optional[str] = None
    monthly_cost: float = 0.0
    region: str = "us-east-1"
    created_date: str = "2025-06-15"
    last_accessed_date: str = "2026-04-01"
    optimal_action: str = "keep"

    def __post_init__(self) -> None:
        if self.monthly_cost == 0.0:
            self.monthly_cost = round(
                self.size_gb * VOLUME_PRICING.get(self.volume_type, 0.08), 2
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "volume_id": self.volume_id,
            "size_gb": self.size_gb,
            "volume_type": self.volume_type,
            "state": self.state,
            "attached_instance_id": self.attached_instance_id,
            "monthly_cost": self.monthly_cost,
            "region": self.region,
            "created_date": self.created_date,
            "last_accessed_date": self.last_accessed_date,
        }


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


# ── Task 1: Easy — Cleanup unused volumes ──────────────────────────────────

def _build_easy_infra() -> CloudState:
    instances = [
        Instance("i-001", "t3.medium", "running", "web-frontend", 45, 72, 40, 30.09,
                 tags={"team": "frontend", "env": "prod"}, network_io_gbps=0.8, uptime_days=220),
        Instance("i-002", "m5.large", "running", "api-server", 62, 85, 55, 69.12,
                 tags={"team": "backend", "env": "prod"}, network_io_gbps=2.1, uptime_days=180),
        Instance("i-003", "t3.small", "running", "monitoring", 28, 41, 30, 15.04,
                 tags={"team": "devops", "env": "prod"}, network_io_gbps=0.3, uptime_days=365),
        Instance("i-004", "c5.large", "running", "batch-worker", 55, 78, 42, 61.20,
                 tags={"team": "data", "env": "prod"}, network_io_gbps=1.5, uptime_days=90),
        Instance("i-005", "t3.micro", "running", "bastion", 12, 25, 15, 7.49,
                 tags={"team": "devops", "env": "prod"}, network_io_gbps=0.1, uptime_days=400),
        Instance("i-006", "m5.xlarge", "running", "database", 70, 92, 68, 138.24,
                 tags={"team": "backend", "env": "prod"}, network_io_gbps=3.0, uptime_days=300),
    ]
    volumes = [
        Volume("vol-001", 100, "gp3", "in-use", "i-001"),
        Volume("vol-002", 500, "gp3", "in-use", "i-002"),
        Volume("vol-003",  50, "gp3", "available", None, optimal_action="delete",
               created_date="2025-01-10", last_accessed_date="2025-03-15"),
        Volume("vol-004", 200, "gp2", "in-use", "i-003"),
        Volume("vol-005", 100, "gp3", "in-use", "i-004"),
        Volume("vol-006", 250, "io1", "available", None, optimal_action="delete",
               created_date="2025-04-20", last_accessed_date="2025-08-01"),
        Volume("vol-007",  50, "gp3", "in-use", "i-005"),
        Volume("vol-008", 1000, "gp3", "available", None, optimal_action="delete",
               created_date="2024-11-05", last_accessed_date="2025-02-28"),
        Volume("vol-009", 500, "gp2", "in-use", "i-006"),
        Volume("vol-010",  75, "gp2", "available", None, optimal_action="delete",
               created_date="2025-07-18", last_accessed_date="2025-09-30"),
    ]
    return CloudState(instances, volumes)


# ── Task 2: Medium — Right-size over-provisioned ───────────────────────────

def _build_medium_infra() -> CloudState:
    instances = [
        Instance("i-101", "m5.2xlarge", "running", "legacy-api", 8, 15, 12, 276.48,
                 tags={"team": "backend", "env": "prod"}, uptime_days=500,
                 optimal_action="resize", optimal_type="m5.large"),
        Instance("i-102", "c5.2xlarge", "running", "test-runner", 5, 10, 8, 244.80,
                 tags={"team": "qa", "env": "staging"}, uptime_days=300,
                 optimal_action="resize", optimal_type="c5.large"),
        Instance("i-103", "r5.xlarge", "running", "cache", 65, 88, 72, 181.44,
                 tags={"team": "backend", "env": "prod"}, uptime_days=150),
        Instance("i-104", "m5.xlarge", "running", "ml-pipeline", 12, 22, 18, 138.24,
                 tags={"team": "ml", "env": "prod"}, uptime_days=200,
                 optimal_action="resize", optimal_type="m5.large"),
        Instance("i-105", "t3.large", "running", "staging-web", 7, 14, 10, 60.12,
                 tags={"team": "frontend", "env": "staging"}, uptime_days=120,
                 optimal_action="resize", optimal_type="t3.small"),
        Instance("i-106", "c5.xlarge", "running", "data-proc", 58, 80, 45, 122.40,
                 tags={"team": "data", "env": "prod"}, uptime_days=250),
        Instance("i-107", "m5.xlarge", "running", "analytics", 9, 18, 14, 138.24,
                 tags={"team": "analytics", "env": "prod"}, uptime_days=180,
                 optimal_action="resize", optimal_type="m5.large"),
        Instance("i-108", "t3.medium", "running", "log-collector", 42, 65, 38, 30.09,
                 tags={"team": "devops", "env": "prod"}, uptime_days=400),
    ]
    volumes = [
        Volume("vol-101", 200, "gp3", "in-use", "i-101"),
        Volume("vol-102", 500, "gp3", "in-use", "i-103"),
        Volume("vol-103", 100, "gp3", "in-use", "i-106"),
        Volume("vol-104", 300, "gp3", "in-use", "i-108"),
    ]
    return CloudState(instances, volumes)


# ── Task 3: Hard — Full cost optimization (enhanced) ──────────────────────

def _build_hard_infra() -> CloudState:
    instances = [
        Instance("i-201", "m5.2xlarge", "running", "prod-api", 55, 82, 60, 276.48,
                 cpu_history_7d=[52, 54, 55, 58, 55, 53, 56],
                 dependencies=["staging-api"],
                 tags={"team": "backend", "env": "prod", "tier": "critical"},
                 region="us-east-1", network_io_gbps=4.5, uptime_days=365),
        Instance("i-202", "m5.xlarge", "running", "staging-api", 3, 8, 5, 138.24,
                 cpu_history_7d=[3, 2, 4, 3, 3, 2, 4],
                 tags={"team": "backend", "env": "staging"},
                 uptime_days=90, optimal_action="terminate"),
        Instance("i-203", "c5.2xlarge", "running", "ml-training", 6, 12, 9, 244.80,
                 cpu_history_7d=[5, 6, 8, 5, 7, 6, 5],
                 tags={"team": "ml", "env": "dev"},
                 uptime_days=60, optimal_action="resize", optimal_type="c5.large"),
        Instance("i-204", "t3.large", "running", "dev-server", 2, 5, 3, 60.12,
                 cpu_history_7d=[2, 1, 3, 2, 2, 1, 2],
                 tags={"team": "frontend", "env": "dev"},
                 uptime_days=45, optimal_action="terminate"),
        Instance("i-205", "r5.xlarge", "running", "prod-db", 72, 95, 80, 181.44,
                 cpu_history_7d=[70, 72, 75, 72, 68, 73, 74],
                 dependencies=["prod-api", "analytics", "etl-pipeline"],
                 tags={"team": "backend", "env": "prod", "tier": "critical"},
                 region="us-east-1", network_io_gbps=5.0, uptime_days=600),
        Instance("i-206", "m5.xlarge", "running", "analytics", 11, 20, 15, 138.24,
                 cpu_history_7d=[10, 12, 11, 13, 10, 9, 12],
                 tags={"team": "analytics", "env": "prod"},
                 uptime_days=200, optimal_action="resize", optimal_type="m5.large"),
        Instance("i-207", "c5.xlarge", "running", "batch-jobs", 45, 70, 40, 122.40,
                 cpu_history_7d=[42, 45, 48, 50, 43, 40, 47],
                 tags={"team": "data", "env": "prod"},
                 region="us-west-2", network_io_gbps=2.0, uptime_days=300),
        Instance("i-208", "t3.medium", "running", "monitoring", 35, 55, 30, 30.09,
                 cpu_history_7d=[33, 35, 36, 38, 34, 32, 35],
                 dependencies=["prod-api", "batch-jobs", "etl-pipeline"],
                 tags={"team": "devops", "env": "prod", "tier": "critical"},
                 uptime_days=500),
        Instance("i-209", "m5.2xlarge", "running", "old-reports", 1, 3, 2, 276.48,
                 cpu_history_7d=[1, 1, 2, 1, 1, 0, 1],
                 tags={"team": "analytics", "env": "prod", "status": "deprecated"},
                 uptime_days=700, optimal_action="terminate"),
        Instance("i-210", "t3.large", "running", "qa-server", 4, 9, 6, 60.12,
                 cpu_history_7d=[4, 3, 5, 4, 4, 3, 5],
                 tags={"team": "qa", "env": "staging"},
                 uptime_days=120, optimal_action="terminate"),
        Instance("i-211", "c5.xlarge", "running", "etl-pipeline", 15, 28, 20, 122.40,
                 cpu_history_7d=[14, 15, 16, 18, 14, 13, 16],
                 tags={"team": "data", "env": "prod"},
                 uptime_days=250, optimal_action="resize", optimal_type="c5.large"),
        Instance("i-212", "m5.xlarge", "running", "search-svc", 52, 78, 55, 138.24,
                 cpu_history_7d=[50, 52, 55, 53, 51, 49, 54],
                 dependencies=["prod-api", "batch-jobs"],
                 tags={"team": "backend", "env": "prod"},
                 region="us-east-1", network_io_gbps=3.0, uptime_days=400),
        Instance("i-213", "r5.large", "running", "cache-layer", 60, 85, 65, 90.72,
                 cpu_history_7d=[58, 60, 62, 63, 59, 57, 61],
                 dependencies=["prod-api"],
                 tags={"team": "backend", "env": "prod"},
                 network_io_gbps=4.0, uptime_days=300),
        Instance("i-214", "t3.small", "running", "cron-jobs", 8, 18, 10, 15.04,
                 cpu_history_7d=[7, 8, 9, 8, 7, 6, 9],
                 tags={"team": "devops", "env": "prod"},
                 uptime_days=200),
        Instance("i-215", "m5.xlarge", "running", "log-agg", 10, 19, 13, 138.24,
                 cpu_history_7d=[9, 10, 11, 12, 10, 9, 11],
                 tags={"team": "devops", "env": "prod"},
                 uptime_days=150, optimal_action="resize", optimal_type="m5.large"),
        # TRAP: Low avg but massive weekend spike in 7d history
        Instance("i-216", "m5.xlarge", "running", "weekend-batch", 18, 72, 22, 138.24,
                 cpu_history_7d=[5, 6, 4, 5, 7, 68, 72],
                 tags={"team": "data", "env": "prod"},
                 uptime_days=100),
    ]
    volumes = [
        Volume("vol-201", 500, "gp3", "in-use", "i-201"),
        Volume("vol-202", 200, "gp3", "available", None, optimal_action="delete",
               created_date="2025-02-10", last_accessed_date="2025-05-01"),
        Volume("vol-203", 1000, "io1", "in-use", "i-205"),
        Volume("vol-204", 100, "gp3", "available", None, optimal_action="delete",
               created_date="2025-08-15", last_accessed_date="2025-11-20"),
        Volume("vol-205", 300, "gp2", "in-use", "i-207"),
        Volume("vol-206",  50, "gp3", "in-use", "i-208"),
        Volume("vol-207", 500, "gp3", "available", None, optimal_action="delete",
               created_date="2024-12-01", last_accessed_date="2025-04-10"),
        Volume("vol-208", 200, "gp3", "in-use", "i-212"),
        Volume("vol-209", 100, "gp2", "available", None, optimal_action="delete",
               created_date="2025-06-20", last_accessed_date="2025-10-15"),
        Volume("vol-210", 750, "gp3", "in-use", "i-213"),
        Volume("vol-211", 100, "gp3", "in-use", "i-201"),
        Volume("vol-212", 150, "gp3", "available", None, optimal_action="delete",
               created_date="2025-03-05", last_accessed_date="2025-07-22"),
    ]
    return CloudState(instances, volumes)


# ── Task 4: Medium-Hard — Spot Instance Migration ─────────────────────────

def _build_spot_migration_infra() -> CloudState:
    """
    10 instances. Agent must identify which ones are safe to convert to spot
    pricing. Rules:
    - Only stateless, fault-tolerant workloads should go to spot
    - Production-critical and stateful instances must stay on-demand
    - Dependencies mean the instance can't tolerate interruption
    - High memory (r5) has higher interruption risk
    """
    instances = [
        # Safe for spot: stateless batch workers, dev/staging, no deps
        Instance("i-301", "c5.xlarge", "running", "batch-etl", 40, 65, 35, 122.40,
                 tags={"team": "data", "env": "prod", "stateful": "false"},
                 spot_eligible=True, uptime_days=200, network_io_gbps=1.5,
                 optimal_action="convert_to_spot"),
        Instance("i-302", "m5.xlarge", "running", "dev-api", 15, 30, 20, 138.24,
                 tags={"team": "backend", "env": "dev", "stateful": "false"},
                 spot_eligible=True, uptime_days=60, network_io_gbps=0.5,
                 optimal_action="convert_to_spot"),
        Instance("i-303", "c5.2xlarge", "running", "ci-runner", 35, 55, 25, 244.80,
                 tags={"team": "devops", "env": "staging", "stateful": "false"},
                 spot_eligible=True, uptime_days=150, network_io_gbps=1.0,
                 optimal_action="convert_to_spot"),
        Instance("i-304", "m5.large", "running", "test-worker", 20, 40, 18, 69.12,
                 tags={"team": "qa", "env": "staging", "stateful": "false"},
                 spot_eligible=True, uptime_days=90, network_io_gbps=0.3,
                 optimal_action="convert_to_spot"),
        # NOT safe for spot: prod, stateful, deps, critical
        Instance("i-305", "r5.xlarge", "running", "prod-db-replica", 55, 80, 70, 181.44,
                 dependencies=["prod-api"],
                 tags={"team": "backend", "env": "prod", "stateful": "true", "tier": "critical"},
                 spot_eligible=False, uptime_days=400, network_io_gbps=4.0),
        Instance("i-306", "m5.2xlarge", "running", "prod-api-main", 60, 85, 55, 276.48,
                 dependencies=["search-svc", "analytics"],
                 tags={"team": "backend", "env": "prod", "stateful": "false", "tier": "critical"},
                 spot_eligible=False, uptime_days=365, network_io_gbps=5.0),
        Instance("i-307", "m5.xlarge", "running", "session-store", 45, 70, 60, 138.24,
                 tags={"team": "backend", "env": "prod", "stateful": "true"},
                 spot_eligible=False, uptime_days=300, network_io_gbps=2.5),
        # TRAP: looks like a batch job but has dependencies
        Instance("i-308", "c5.xlarge", "running", "data-sync", 30, 50, 25, 122.40,
                 dependencies=["prod-db-replica"],
                 tags={"team": "data", "env": "prod", "stateful": "false"},
                 spot_eligible=False, uptime_days=180, network_io_gbps=3.0),
        # TRAP: dev env but stateful (local DB)
        Instance("i-309", "m5.large", "running", "dev-db", 10, 20, 50, 69.12,
                 tags={"team": "backend", "env": "dev", "stateful": "true"},
                 spot_eligible=False, uptime_days=100, network_io_gbps=0.5),
        Instance("i-310", "t3.large", "running", "load-test", 5, 12, 8, 60.12,
                 tags={"team": "qa", "env": "staging", "stateful": "false"},
                 spot_eligible=True, uptime_days=30, network_io_gbps=0.2,
                 optimal_action="convert_to_spot"),
    ]
    volumes = [
        Volume("vol-301", 500, "gp3", "in-use", "i-305"),
        Volume("vol-302", 200, "gp3", "in-use", "i-306"),
        Volume("vol-303", 100, "gp3", "in-use", "i-307"),
    ]
    return CloudState(instances, volumes)


# ── Task 5: Expert — Reserved Instance Planning ───────────────────────────

def _build_ri_planning_infra() -> CloudState:
    """
    12 instances. Agent must decide which long-running, stable-usage instances
    should be converted to 1-year Reserved Instances. Rules:
    - Only worth it if instance has been running 180+ days with stable usage
    - Must have consistent CPU (low variance in 7d history)
    - Production env instances with steady workloads are ideal RI candidates
    - Dev/staging/temporary workloads should stay on-demand
    - High variance = unpredictable, don't commit
    - Some instances are TRAPS: long-running but about to be decommissioned
    """
    instances = [
        # Good RI candidates: long-running, stable, prod
        Instance("i-401", "m5.2xlarge", "running", "core-api", 58, 72, 55, 276.48,
                 cpu_history_7d=[57, 58, 59, 57, 58, 56, 58],
                 tags={"team": "backend", "env": "prod", "tier": "critical"},
                 ri_eligible=True, uptime_days=600, network_io_gbps=4.0,
                 optimal_action="purchase_ri"),
        Instance("i-402", "c5.xlarge", "running", "data-proc", 50, 65, 40, 122.40,
                 cpu_history_7d=[49, 50, 52, 51, 50, 48, 51],
                 tags={"team": "data", "env": "prod"},
                 ri_eligible=True, uptime_days=400, network_io_gbps=2.0,
                 optimal_action="purchase_ri"),
        Instance("i-403", "r5.xlarge", "running", "prod-cache", 62, 78, 70, 181.44,
                 cpu_history_7d=[60, 62, 63, 61, 62, 60, 63],
                 tags={"team": "backend", "env": "prod"},
                 ri_eligible=True, uptime_days=500, network_io_gbps=3.5,
                 optimal_action="purchase_ri"),
        Instance("i-404", "m5.xlarge", "running", "search-index", 45, 60, 50, 138.24,
                 cpu_history_7d=[44, 45, 46, 44, 45, 43, 46],
                 tags={"team": "backend", "env": "prod"},
                 ri_eligible=True, uptime_days=350, network_io_gbps=2.5,
                 optimal_action="purchase_ri"),
        # NOT good RI candidates
        Instance("i-405", "t3.large", "running", "staging-app", 20, 40, 15, 60.12,
                 cpu_history_7d=[15, 20, 35, 10, 25, 18, 30],
                 tags={"team": "frontend", "env": "staging"},
                 ri_eligible=False, uptime_days=90),
        Instance("i-406", "m5.large", "running", "experiment-ml", 30, 55, 25, 69.12,
                 cpu_history_7d=[10, 50, 15, 45, 20, 55, 12],
                 tags={"team": "ml", "env": "dev"},
                 ri_eligible=False, uptime_days=45, network_io_gbps=1.0),
        Instance("i-407", "c5.large", "running", "temp-migration", 40, 60, 35, 61.20,
                 cpu_history_7d=[38, 40, 42, 41, 40, 39, 41],
                 tags={"team": "devops", "env": "prod", "status": "migrating"},
                 ri_eligible=False, uptime_days=200),
        # TRAP: Long-running + stable BUT tagged for decommission
        Instance("i-408", "m5.xlarge", "running", "legacy-billing", 35, 48, 30, 138.24,
                 cpu_history_7d=[34, 35, 36, 35, 34, 33, 36],
                 tags={"team": "billing", "env": "prod", "status": "deprecated",
                       "decommission_date": "2026-06-01"},
                 ri_eligible=False, uptime_days=800),
        # TRAP: Stable CPU but wildly variable memory (database-like)
        Instance("i-409", "r5.large", "running", "analytics-db", 25, 35, 85, 90.72,
                 cpu_history_7d=[24, 25, 26, 25, 24, 23, 26],
                 tags={"team": "analytics", "env": "prod"},
                 ri_eligible=True, uptime_days=300, network_io_gbps=2.0,
                 optimal_action="purchase_ri"),
        Instance("i-410", "c5.2xlarge", "running", "nightly-build", 8, 90, 15, 244.80,
                 cpu_history_7d=[2, 3, 2, 88, 3, 2, 85],
                 tags={"team": "devops", "env": "prod"},
                 ri_eligible=False, uptime_days=250),
        Instance("i-411", "t3.medium", "running", "dev-tools", 12, 22, 10, 30.09,
                 cpu_history_7d=[10, 12, 14, 11, 13, 10, 12],
                 tags={"team": "devops", "env": "dev"},
                 ri_eligible=False, uptime_days=60),
        Instance("i-412", "m5.xlarge", "running", "report-gen", 42, 58, 40, 138.24,
                 cpu_history_7d=[41, 42, 43, 42, 41, 40, 43],
                 tags={"team": "analytics", "env": "prod"},
                 ri_eligible=True, uptime_days=450, network_io_gbps=1.5,
                 optimal_action="purchase_ri"),
    ]
    volumes = [
        Volume("vol-401", 1000, "io1", "in-use", "i-401"),
        Volume("vol-402", 500, "gp3", "in-use", "i-403"),
        Volume("vol-403", 300, "gp3", "in-use", "i-409"),
    ]
    return CloudState(instances, volumes)


# ── Task 6: Expert+ — Comprehensive Fleet Review ────────────────────────────

def _build_comprehensive_fleet_infra() -> CloudState:
    """
    18 instances + 10 volumes requiring ALL 5 action types.
    Tests the agent's ability to orchestrate volume cleanup, idle termination,
    right-sizing, spot migration, and RI purchases simultaneously — while
    navigating 5 distinct traps.

    Optimal actions: 3 deletes + 3 terminates + 3 resizes + 3 spots + 3 RIs = 15
    Optimal savings: ~$1,152/month
    """
    instances = [
        # ── TO TERMINATE (idle, no deps, avg CPU < 5%) ──────────────────
        Instance("i-501", "m5.xlarge", "running", "old-staging-api", 2, 5, 4, 138.24,
                 cpu_history_7d=[2, 3, 2, 1, 2, 3, 2],
                 tags={"team": "backend", "env": "staging", "stateful": "false"},
                 uptime_days=200, optimal_action="terminate"),
        Instance("i-502", "c5.large", "running", "unused-worker", 1, 3, 2, 61.20,
                 cpu_history_7d=[1, 1, 2, 1, 0, 1, 1],
                 tags={"team": "data", "env": "dev", "stateful": "false"},
                 uptime_days=90, optimal_action="terminate"),
        Instance("i-503", "t3.large", "running", "legacy-test-runner", 3, 7, 5, 60.12,
                 cpu_history_7d=[3, 4, 3, 2, 3, 4, 3],
                 tags={"team": "qa", "env": "staging", "stateful": "false"},
                 uptime_days=150, optimal_action="terminate"),

        # ── TO RESIZE (over-provisioned, safe downgrades) ───────────────
        Instance("i-504", "m5.2xlarge", "running", "backend-api", 10, 18, 14, 276.48,
                 cpu_history_7d=[9, 10, 12, 11, 10, 9, 11],
                 tags={"team": "backend", "env": "prod", "stateful": "false"},
                 uptime_days=180, optimal_action="resize", optimal_type="m5.large"),
        # peak=18%, cap 4→1, effective=72% < 80% ✓
        Instance("i-505", "c5.2xlarge", "running", "data-pipeline", 8, 14, 11, 244.80,
                 cpu_history_7d=[7, 8, 9, 8, 7, 6, 9],
                 tags={"team": "data", "env": "prod", "stateful": "false"},
                 uptime_days=250, optimal_action="resize", optimal_type="c5.large"),
        # peak=14%, cap 4→1, effective=56% < 80% ✓
        Instance("i-506", "m5.xlarge", "running", "internal-tools", 12, 20, 16, 138.24,
                 cpu_history_7d=[11, 12, 13, 12, 11, 10, 13],
                 tags={"team": "devops", "env": "prod", "stateful": "false"},
                 uptime_days=200, optimal_action="resize", optimal_type="m5.large"),
        # peak=20%, cap 2→1, effective=40% < 80% ✓

        # ── TO CONVERT TO SPOT (stateless, non-critical, no deps) ──────
        Instance("i-507", "c5.xlarge", "running", "batch-processor", 40, 65, 35, 122.40,
                 tags={"team": "data", "env": "prod", "stateful": "false"},
                 spot_eligible=True, uptime_days=200, network_io_gbps=1.5,
                 optimal_action="convert_to_spot"),
        Instance("i-508", "m5.large", "running", "ci-builder", 20, 40, 18, 69.12,
                 tags={"team": "devops", "env": "staging", "stateful": "false"},
                 spot_eligible=True, uptime_days=120, network_io_gbps=0.5,
                 optimal_action="convert_to_spot"),
        Instance("i-509", "t3.large", "running", "load-tester", 15, 30, 12, 60.12,
                 tags={"team": "qa", "env": "staging", "stateful": "false"},
                 spot_eligible=True, uptime_days=60, network_io_gbps=0.2,
                 optimal_action="convert_to_spot"),

        # ── TO PURCHASE RI (stable, long-running, production) ──────────
        Instance("i-510", "m5.2xlarge", "running", "core-service", 55, 72, 50, 276.48,
                 cpu_history_7d=[54, 55, 56, 55, 54, 53, 56],
                 dependencies=["api-gateway"],
                 tags={"team": "backend", "env": "prod", "tier": "critical"},
                 ri_eligible=True, uptime_days=500, network_io_gbps=4.0,
                 optimal_action="purchase_ri"),
        Instance("i-511", "c5.xlarge", "running", "api-gateway", 48, 62, 42, 122.40,
                 cpu_history_7d=[47, 48, 49, 48, 47, 46, 49],
                 tags={"team": "backend", "env": "prod"},
                 ri_eligible=True, uptime_days=400, network_io_gbps=2.5,
                 optimal_action="purchase_ri"),
        Instance("i-512", "r5.xlarge", "running", "prod-cache", 60, 75, 68, 181.44,
                 cpu_history_7d=[59, 60, 61, 60, 59, 58, 61],
                 tags={"team": "backend", "env": "prod"},
                 ri_eligible=True, uptime_days=600, network_io_gbps=3.5,
                 optimal_action="purchase_ri"),

        # ── TO KEEP (critical, well-utilized, or correctly configured) ─
        Instance("i-513", "m5.xlarge", "running", "prod-db", 65, 90, 75, 138.24,
                 cpu_history_7d=[63, 65, 68, 66, 64, 62, 67],
                 dependencies=["core-service", "api-gateway", "data-pipeline"],
                 tags={"team": "backend", "env": "prod", "tier": "critical",
                       "stateful": "true"},
                 uptime_days=700, network_io_gbps=5.0),
        Instance("i-514", "r5.large", "running", "session-store", 55, 78, 82, 90.72,
                 cpu_history_7d=[53, 55, 57, 56, 54, 52, 56],
                 tags={"team": "backend", "env": "prod", "stateful": "true"},
                 uptime_days=300, network_io_gbps=2.0),
        Instance("i-515", "c5.large", "running", "monitoring", 45, 68, 40, 61.20,
                 cpu_history_7d=[43, 45, 47, 46, 44, 42, 46],
                 dependencies=["prod-db", "core-service"],
                 tags={"team": "devops", "env": "prod", "tier": "critical"},
                 uptime_days=500),

        # ── TRAPS ──────────────────────────────────────────────────────
        # TRAP 1: Looks idle but has critical dependencies → DON'T terminate
        Instance("i-516", "m5.xlarge", "running", "config-service", 3, 6, 4, 138.24,
                 cpu_history_7d=[3, 3, 4, 3, 2, 3, 3],
                 dependencies=["core-service", "api-gateway", "batch-processor"],
                 tags={"team": "platform", "env": "prod", "tier": "critical"},
                 uptime_days=400),
        # TRAP 2: Long-running + stable but DEPRECATED → DON'T purchase RI
        Instance("i-517", "m5.xlarge", "running", "legacy-billing", 38, 50, 35, 138.24,
                 cpu_history_7d=[37, 38, 39, 38, 37, 36, 39],
                 tags={"team": "billing", "env": "prod",
                       "status": "deprecated", "decommission_date": "2026-08-01"},
                 uptime_days=800),
        # TRAP 3: Low avg CPU but massive weekend spikes → DON'T resize
        Instance("i-518", "m5.xlarge", "running", "weekend-analytics", 14, 70, 18, 138.24,
                 cpu_history_7d=[5, 6, 4, 5, 7, 65, 70],
                 tags={"team": "analytics", "env": "prod", "stateful": "false"},
                 uptime_days=100),
    ]

    volumes = [
        # ── TO DELETE (unattached) ─────────────────────────────────────
        Volume("vol-501", 200, "gp3", "available", None, optimal_action="delete",
               created_date="2025-03-10", last_accessed_date="2025-07-15"),
        Volume("vol-502", 300, "gp2", "available", None, optimal_action="delete",
               created_date="2025-01-20", last_accessed_date="2025-05-01"),
        Volume("vol-503", 100, "gp3", "available", None, optimal_action="delete",
               created_date="2025-06-05", last_accessed_date="2025-10-30"),
        # ── TO KEEP (attached) ─────────────────────────────────────────
        Volume("vol-504", 500, "gp3", "in-use", "i-510"),
        Volume("vol-505", 1000, "io1", "in-use", "i-513"),
        Volume("vol-506", 200, "gp3", "in-use", "i-512"),
        Volume("vol-507", 300, "gp3", "in-use", "i-514"),
        Volume("vol-508", 100, "gp3", "in-use", "i-515"),
        Volume("vol-509", 500, "gp3", "in-use", "i-504"),
        Volume("vol-510", 200, "gp3", "in-use", "i-507"),
    ]

    return CloudState(instances, volumes)


# ── Public API ──────────────────────────────────────────────────────────────

TASK_BUILDERS = {
    "cleanup_unused_volumes":     _build_easy_infra,
    "rightsize_overprovisioned":  _build_medium_infra,
    "full_cost_optimization":     _build_hard_infra,
    "spot_instance_migration":    _build_spot_migration_infra,
    "reserved_instance_planning": _build_ri_planning_infra,
    "comprehensive_fleet_review": _build_comprehensive_fleet_infra,
}


def build_infrastructure(task_id: str) -> CloudState:
    builder = TASK_BUILDERS.get(task_id)
    if builder is None:
        raise ValueError(
            f"Unknown task_id={task_id!r}. "
            f"Choose from: {list(TASK_BUILDERS.keys())}"
        )
    return builder()
