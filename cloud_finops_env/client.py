"""
Cloud FinOps Environment — Client
====================================
EnvClient subclass for connecting to the Cloud FinOps server.
"""

from __future__ import annotations

from typing import Any, Dict

from cloud_finops_env.models import (
    CloudFinOpsAction,
    CloudFinOpsObservation,
    CloudFinOpsState,
)

try:
    from openenv.core.env_client import EnvClient
    from openenv.core.types import StepResult

    class CloudFinOpsEnv(EnvClient[CloudFinOpsAction, CloudFinOpsObservation, CloudFinOpsState]):
        """
        Async client for the Cloud FinOps environment.

        Usage:
            # Connect to running server
            env = CloudFinOpsEnv(base_url="http://localhost:7860")

            # Or spin up from Docker image
            env = await CloudFinOpsEnv.from_docker_image("cloud-finops-env")
        """

        def _step_payload(self, action: CloudFinOpsAction) -> Dict[str, Any]:
            return action.model_dump()

        def _parse_result(self, data: Dict[str, Any]) -> StepResult[CloudFinOpsObservation]:
            obs_data = data.get("observation", data)
            obs = CloudFinOpsObservation(**obs_data)
            return StepResult(
                observation=obs,
                reward=data.get("reward", obs.reward),
                done=data.get("done", obs.done),
            )

        def _parse_state(self, data: Dict[str, Any]) -> CloudFinOpsState:
            return CloudFinOpsState(**data)

except ImportError:
    # Lightweight fallback when openenv-core is not installed
    import asyncio
    from dataclasses import dataclass

    import requests

    @dataclass
    class StepResult:
        observation: CloudFinOpsObservation
        reward: float | None = None
        done: bool = False

    class CloudFinOpsEnv:  # type: ignore[no-redef]
        """Lightweight sync/async HTTP client fallback."""

        def __init__(self, base_url: str = "http://localhost:7860") -> None:
            self.base_url = base_url.rstrip("/")

        async def reset(self, **kwargs: Any) -> StepResult:
            resp = requests.post(f"{self.base_url}/reset", json=kwargs, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            obs = CloudFinOpsObservation(**data.get("observation", data))
            return StepResult(
                observation=obs,
                reward=data.get("reward"),
                done=data.get("done", False),
            )

        async def step(self, action: CloudFinOpsAction, **kwargs: Any) -> StepResult:
            payload = {"action": action.model_dump()}
            resp = requests.post(f"{self.base_url}/step", json=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            obs = CloudFinOpsObservation(**data.get("observation", data))
            return StepResult(
                observation=obs,
                reward=data.get("reward"),
                done=data.get("done", False),
            )

        async def state(self) -> CloudFinOpsState:
            resp = requests.get(f"{self.base_url}/state", timeout=10)
            resp.raise_for_status()
            return CloudFinOpsState(**resp.json())

        async def close(self) -> None:
            pass
