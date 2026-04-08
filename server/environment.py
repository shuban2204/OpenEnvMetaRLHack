"""
Cloud FinOps Environment — Core Logic
=======================================
Implements the OpenEnv Environment interface: reset(), step(), state.
"""

from __future__ import annotations

import os
import uuid
from typing import Any, Dict, List, Optional

from cloud_finops_env.models import (
    DOWNGRADE_PATHS,
    INSTANCE_PRICING,
    RI_PRICING,
    SPOT_PRICING,
    CloudFinOpsAction,
    CloudFinOpsObservation,
    CloudFinOpsState,
)
from server.simulator import (
    CloudState,
    build_infrastructure,
)
from server.tasks import (
    TASKS,
    TaskDef,
    compute_step_reward,
    grade_episode,
)


class CloudFinOpsEnvironment:
    """
    Simulated cloud infrastructure environment for cost-optimization agents.

    The agent observes instances and volumes with utilization metrics, then
    takes actions (terminate, delete, resize, skip, submit) to reduce monthly
    spend while respecting SLA constraints.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = False

    def __init__(self) -> None:
        self._task_def: Optional[TaskDef] = None
        self._initial_state: Optional[CloudState] = None
        self._current_state: Optional[CloudState] = None
        self._episode_id: str = ""
        self._step_count: int = 0
        self._savings: float = 0.0
        self._violations: List[str] = []
        self._action_history: List[str] = []
        self._done: bool = False
        self._initial_spend: float = 0.0
        self._last_action_error: Optional[str] = None

    # ── reset ───────────────────────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> CloudFinOpsObservation:
        task_id = kwargs.get("task_id") or os.getenv(
            "CLOUD_FINOPS_TASK", "cleanup_unused_volumes"
        )
        if task_id not in TASKS:
            task_id = "cleanup_unused_volumes"

        self._task_def = TASKS[task_id]
        self._initial_state = build_infrastructure(task_id)
        self._current_state = self._initial_state.deep_copy()
        self._episode_id = episode_id or str(uuid.uuid4())
        self._step_count = 0
        self._savings = 0.0
        self._violations = []
        self._action_history = []
        self._done = False
        self._initial_spend = self._initial_state.total_monthly_spend()
        self._last_action_error = None

        return self._build_observation(reward=None)

    # ── step ────────────────────────────────────────────────────────────

    def step(
        self,
        action: CloudFinOpsAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> CloudFinOpsObservation:
        if self._done:
            return self._build_observation(reward=0.0)

        assert self._current_state is not None
        assert self._initial_state is not None
        assert self._task_def is not None

        self._step_count += 1
        a_type = action.action_type
        target = action.target_id
        new_type = action.new_type

        # Record action
        action_str = a_type
        if target:
            action_str += f"({target}"
            if new_type:
                action_str += f", {new_type}"
            action_str += ")"
        self._action_history.append(action_str)

        # ── submit ──────────────────────────────────────────────────────
        if a_type == "submit":
            self._done = True
            final_score = grade_episode(
                self._task_def.task_id,
                self._savings,
                self._violations,
                self._initial_state,
                self._current_state,
            )
            return self._build_observation(reward=final_score)

        # ── compute reward ──────────────────────────────────────────────
        reward, savings_delta, violation = compute_step_reward(
            self._task_def.task_id,
            a_type,
            target,
            new_type,
            self._current_state,
            self._initial_state,
        )

        self._last_action_error = violation
        if violation:
            self._violations.append(violation)

        # ── mutate state (realistic: actions execute even if sub-optimal) ─
        if a_type == "delete_volume":
            vol = next(
                (v for v in self._current_state.volumes if v.volume_id == target),
                None,
            )
            # AWS blocks deleting attached volumes; unattached ones always delete
            if vol and vol.state == "available":
                vol.state = "deleted"
                self._savings += vol.monthly_cost

        elif a_type == "terminate_instance":
            inst = next(
                (i for i in self._current_state.instances if i.instance_id == target),
                None,
            )
            if inst and inst.state == "running":
                if not inst.dependencies:
                    # Termination executes — even if sub-optimal (irreversible!)
                    inst.state = "terminated"
                    self._savings += inst.monthly_cost
                # If dependencies exist, cloud provider blocks the termination
                # (simulated by org-level termination protection policy)

        elif a_type == "resize_instance":
            inst = next(
                (i for i in self._current_state.instances if i.instance_id == target),
                None,
            )
            if inst and inst.state == "running":
                allowed = DOWNGRADE_PATHS.get(inst.instance_type, [])
                if new_type in allowed and new_type in INSTANCE_PRICING:
                    # Resize always executes — SLA violation is a consequence,
                    # not a blocker (app may crash under peak load)
                    old_cost = inst.monthly_cost
                    inst.instance_type = new_type
                    inst.monthly_cost = INSTANCE_PRICING[new_type]
                    self._savings += round(old_cost - inst.monthly_cost, 2)

        elif a_type == "convert_to_spot":
            inst = next(
                (i for i in self._current_state.instances if i.instance_id == target),
                None,
            )
            if inst and inst.state == "running" and inst.pricing_model != "spot":
                if not inst.dependencies:
                    spot_cost = SPOT_PRICING.get(inst.instance_type, inst.monthly_cost)
                    old_cost = inst.monthly_cost
                    inst.pricing_model = "spot"
                    inst.monthly_cost = spot_cost
                    self._savings += round(old_cost - spot_cost, 2)

        elif a_type == "purchase_ri":
            inst = next(
                (i for i in self._current_state.instances if i.instance_id == target),
                None,
            )
            if inst and inst.state == "running" and inst.pricing_model != "reserved":
                ri_cost = RI_PRICING.get(inst.instance_type, inst.monthly_cost)
                old_cost = inst.monthly_cost
                inst.pricing_model = "reserved"
                inst.monthly_cost = ri_cost
                self._savings += round(old_cost - ri_cost, 2)

        # ── check episode end ───────────────────────────────────────────
        if self._step_count >= self._task_def.max_steps:
            self._done = True
            final_score = grade_episode(
                self._task_def.task_id,
                self._savings,
                self._violations,
                self._initial_state,
                self._current_state,
            )
            reward = final_score

        return self._build_observation(reward=reward)

    # ── state ───────────────────────────────────────────────────────────

    @property
    def state(self) -> CloudFinOpsState:
        return CloudFinOpsState(
            episode_id=self._episode_id,
            step_count=self._step_count,
            task_id=self._task_def.task_id if self._task_def else "",
            savings_achieved=self._savings,
            violations_count=len(self._violations),
        )

    # ── close ───────────────────────────────────────────────────────────

    def close(self) -> None:
        pass

    # ── metadata ────────────────────────────────────────────────────────

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "name": "cloud_finops_env",
            "description": (
                "Cloud FinOps & Infrastructure Right-Sizing Environment. "
                "An AI agent acts as a Cloud Financial Engineer, optimizing "
                "cloud spend by deleting unused volumes, terminating idle "
                "instances, and right-sizing over-provisioned resources."
            ),
            "version": "1.0.0",
            "tasks": list(TASKS.keys()),
        }

    # ── helpers ─────────────────────────────────────────────────────────

    def _build_observation(self, reward: Optional[float]) -> CloudFinOpsObservation:
        assert self._task_def is not None
        cs = self._current_state
        assert cs is not None

        active_instances = [
            i.to_dict() for i in cs.instances if i.state != "terminated"
        ]
        active_volumes = [
            v.to_dict() for v in cs.volumes if v.state != "deleted"
        ]

        # Build resize options for currently running instances
        resize_opts: Dict[str, List[str]] = {}
        for inst in cs.instances:
            if inst.state == "running" and inst.instance_type in DOWNGRADE_PATHS:
                resize_opts[inst.instance_type] = DOWNGRADE_PATHS[inst.instance_type]

        # Build spot/RI pricing for active instance types
        active_types = set(
            i.instance_type for i in cs.instances if i.state == "running"
        )
        spot_prices = {t: SPOT_PRICING[t] for t in active_types if t in SPOT_PRICING}
        ri_prices = {t: RI_PRICING[t] for t in active_types if t in RI_PRICING}

        return CloudFinOpsObservation(
            done=self._done,
            reward=reward,
            instances=active_instances,
            volumes=active_volumes,
            current_monthly_spend=cs.total_monthly_spend(),
            savings_achieved=round(self._savings, 2),
            violations=list(self._violations),
            task_description=self._task_def.description,
            task_id=self._task_def.task_id,
            step_number=self._step_count,
            max_steps=self._task_def.max_steps,
            resize_options=resize_opts,
            spot_pricing=spot_prices,
            ri_pricing=ri_prices,
            action_history=list(self._action_history),
            last_action_error=self._last_action_error,
        )
