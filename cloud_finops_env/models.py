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

# Relative compute capacity within each family (used for SLA-violation checks)
INSTANCE_CAPACITY: Dict[str, float] = {
    "t3.micro":   0.5,
    "t3.small":   1.0,
    "t3.medium":  2.0,
    "t3.large":   4.0,
    "m5.large":   1.0,
    "m5.xlarge":  2.0,
    "m5.2xlarge": 4.0,
    "c5.large":   1.0,
    "c5.xlarge":  2.0,
    "c5.2xlarge": 4.0,
    "r5.large":   1.0,
    "r5.xlarge":  2.0,
}

# Which smaller types each type can be downsized to (same family)
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
    "gp3":  0.08,
    "gp2":  0.10,
    "io1":  0.125,
}


# ---------------------------------------------------------------------------
# Sub-models used inside observations
# ---------------------------------------------------------------------------
class InstanceInfo(BaseModel):
    instance_id: str
    instance_type: str
    state: str                                    # "running" | "stopped" | "terminated"
    app_name: str
    cpu_avg_percent: float
    cpu_peak_percent: float
    memory_avg_percent: float
    monthly_cost: float
    cpu_history_7d: List[float] = Field(default_factory=list)
    dependencies: List[str] = Field(default_factory=list)   # apps that depend on THIS instance


class VolumeInfo(BaseModel):
    volume_id: str
    size_gb: int
    volume_type: str                              # "gp3" | "gp2" | "io1"
    state: str                                    # "in-use" | "available"
    attached_instance_id: Optional[str] = None
    monthly_cost: float


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
        - "skip"                : do nothing this step
        - "submit"              : signal done — triggers final scoring
    """
    action_type: str = "skip"
    target_id: str = ""
    new_type: str = ""        # only relevant for resize_instance


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
            "terminate_instance", "delete_volume",
            "resize_instance", "skip", "submit",
        ]
    )
    resize_options: Dict[str, List[str]] = Field(default_factory=dict)
    action_history: List[str] = Field(default_factory=list)
    last_action_error: Optional[str] = None


# ---------------------------------------------------------------------------
# OpenEnv State
# ---------------------------------------------------------------------------
class CloudFinOpsState(_BaseState):
    task_id: str = ""
    savings_achieved: float = 0.0
    violations_count: int = 0
