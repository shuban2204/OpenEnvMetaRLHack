"""
Cloud FinOps Environment - Pydantic Models
===========================================
Typed Action, Observation, and State models for the Cloud FinOps
infrastructure right-sizing environment.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Try importing OpenEnv base classes; fall back to plain Pydantic models
# ---------------------------------------------------------------------------
try:
    from openenv.core.types import Action as _BaseAction
    from openenv.core.types import Observation as _BaseObservation
    from openenv.core.types import State as _BaseState
except ImportError:
    class _BaseAction(BaseModel):  # type: ignore[no-redef]
        metadata: Dict[str, Any] = Field(default_factory=dict)

    class _BaseObservation(BaseModel):  # type: ignore[no-redef]
        done: bool = False
        reward: Union[float, None] = None
        metadata: Dict[str, Any] = Field(default_factory=dict)

    class _BaseState(BaseModel):  # type: ignore[no-redef]
        episode_id: Optional[str] = None
        step_count: int = 0


# ---------------------------------------------------------------------------
# Instance pricing (monthly USD, approximate real AWS pricing)
# ---------------------------------------------------------------------------
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

# Spot pricing (typically 60-70% discount)
SPOT_PRICING: Dict[str, float] = {
    "t3.micro":    2.25, "t3.small":   4.51, "t3.medium":  9.03,
    "t3.large":   18.04,
    "m5.large":   20.74, "m5.xlarge": 41.47, "m5.2xlarge": 82.94,
    "c5.large":   18.36, "c5.xlarge": 36.72, "c5.2xlarge": 73.44,
    "r5.large":   27.22, "r5.xlarge": 54.43,
}

# 1-year Reserved Instance pricing (typically 30-40% discount)
RI_PRICING: Dict[str, float] = {
    "t3.micro":    4.87, "t3.small":   9.78, "t3.medium": 19.56,
    "t3.large":   39.08,
    "m5.large":   44.93, "m5.xlarge": 89.86, "m5.2xlarge":179.71,
    "c5.large":   39.78, "c5.xlarge": 79.56, "c5.2xlarge":159.12,
    "r5.large":   58.97, "r5.xlarge": 117.94,
}

# Relative compute capacity within each family
INSTANCE_CAPACITY: Dict[str, float] = {
    "t3.micro":   0.5, "t3.small":  1.0, "t3.medium": 2.0, "t3.large":  4.0,
    "m5.large":   1.0, "m5.xlarge": 2.0, "m5.2xlarge":4.0,
    "c5.large":   1.0, "c5.xlarge": 2.0, "c5.2xlarge":4.0,
    "r5.large":   1.0, "r5.xlarge": 2.0,
}

# Downgrade paths within each family
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

# Volume pricing per GB/month
VOLUME_PRICING: Dict[str, float] = {
    "gp3": 0.08, "gp2": 0.10, "io1": 0.125,
}

# Spot interruption risk by instance type family
SPOT_INTERRUPTION_RISK: Dict[str, str] = {
    "t3": "low", "m5": "medium", "c5": "medium", "r5": "high",
}


# ---------------------------------------------------------------------------
# Sub-models used inside observations
# ---------------------------------------------------------------------------
class InstanceInfo(BaseModel):
    instance_id: str
    instance_type: str
    state: str                    # "running" | "stopped" | "terminated"
    app_name: str
    cpu_avg_percent: float
    cpu_peak_percent: float
    memory_avg_percent: float
    monthly_cost: float
    cpu_history_7d: List[float] = Field(default_factory=list)
    dependencies: List[str] = Field(default_factory=list)
    # Enriched fields
    tags: Dict[str, str] = Field(default_factory=dict)
    region: str = "us-east-1"
    launch_date: str = ""        # ISO date string
    pricing_model: str = "on-demand"  # "on-demand" | "spot" | "reserved"
    spot_eligible: bool = False
    ri_eligible: bool = False
    network_io_gbps: float = 0.0
    uptime_days: int = 0


class VolumeInfo(BaseModel):
    volume_id: str
    size_gb: int
    volume_type: str              # "gp3" | "gp2" | "io1"
    state: str                    # "in-use" | "available"
    attached_instance_id: Optional[str] = None
    monthly_cost: float
    # Enriched fields
    region: str = "us-east-1"
    created_date: str = ""
    last_accessed_date: str = ""


# ---------------------------------------------------------------------------
# OpenEnv Action
# ---------------------------------------------------------------------------
class CloudFinOpsAction(_BaseAction):
    """
    Agent action for the Cloud FinOps environment.

    action_type:
        - "terminate_instance"  : shut down and remove an instance
        - "delete_volume"       : delete a storage volume
        - "resize_instance"     : change an instance to a smaller type
        - "convert_to_spot"     : switch an on-demand instance to spot pricing
        - "purchase_ri"         : buy a 1-year reserved instance commitment
        - "skip"                : do nothing this step
        - "submit"              : signal done — triggers final scoring
    """
    action_type: str = "skip"
    target_id: str = ""
    new_type: str = ""        # for resize_instance


# ---------------------------------------------------------------------------
# OpenEnv Observation
# ---------------------------------------------------------------------------
class CloudFinOpsObservation(_BaseObservation):
    instances: List[Dict[str, Any]] = Field(default_factory=list)
    volumes: List[Dict[str, Any]] = Field(default_factory=list)
    current_monthly_spend: float = 0.0
    savings_achieved: float = 0.0
    violations: List[str] = Field(default_factory=list)
    task_description: str = ""
    task_id: str = ""
    step_number: int = 0
    max_steps: int = 15
    available_actions: List[str] = Field(
        default_factory=lambda: [
            "terminate_instance", "delete_volume", "resize_instance",
            "convert_to_spot", "purchase_ri", "skip", "submit",
        ]
    )
    resize_options: Dict[str, List[str]] = Field(default_factory=dict)
    spot_pricing: Dict[str, float] = Field(default_factory=dict)
    ri_pricing: Dict[str, float] = Field(default_factory=dict)
    action_history: List[str] = Field(default_factory=list)
    last_action_error: Optional[str] = None
    budget_limit: Optional[float] = None


# ---------------------------------------------------------------------------
# OpenEnv State
# ---------------------------------------------------------------------------
class CloudFinOpsState(_BaseState):
    task_id: str = ""
    savings_achieved: float = 0.0
    violations_count: int = 0
