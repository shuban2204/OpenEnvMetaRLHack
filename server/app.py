"""
FastAPI Server — Cloud FinOps Environment
==========================================
Exposes /reset, /step, /state endpoints following the OpenEnv protocol.

Usage:
    uvicorn server.app:app --host 0.0.0.0 --port 7860
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Try to use openenv's create_app; fall back to a hand-rolled FastAPI app
# ---------------------------------------------------------------------------
_USE_OPENENV_CREATE_APP = False

try:
    from openenv.core.env_server import create_app as _oe_create_app

    from cloud_finops_env.models import CloudFinOpsAction, CloudFinOpsObservation
    from server.environment import CloudFinOpsEnvironment

    app = _oe_create_app(
        CloudFinOpsEnvironment,
        CloudFinOpsAction,
        CloudFinOpsObservation,
        env_name="cloud_finops_env",
    )
    _USE_OPENENV_CREATE_APP = True
except Exception:
    app = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Manual FastAPI fallback (also used as extra endpoints when create_app works)
# ---------------------------------------------------------------------------
if not _USE_OPENENV_CREATE_APP:
    from cloud_finops_env.models import CloudFinOpsAction, CloudFinOpsObservation
    from server.environment import CloudFinOpsEnvironment

    app = FastAPI(
        title="Cloud FinOps Environment",
        description="OpenEnv environment for cloud infrastructure cost optimization",
        version="1.0.0",
    )

    _env = CloudFinOpsEnvironment()

    class ResetRequest(BaseModel):
        seed: Optional[int] = None
        episode_id: Optional[str] = None
        task_id: Optional[str] = None

    class StepRequest(BaseModel):
        action: Dict[str, Any]
        timeout_s: Optional[float] = None
        request_id: Optional[str] = None

    @app.post("/reset")
    async def reset_endpoint(request: ResetRequest = ResetRequest()) -> Dict[str, Any]:
        obs = _env.reset(
            seed=request.seed,
            episode_id=request.episode_id,
            task_id=request.task_id,
        )
        obs_dict = obs.model_dump()
        return {
            "observation": obs_dict,
            "reward": obs.reward,
            "done": obs.done,
        }

    @app.post("/step")
    async def step_endpoint(request: StepRequest) -> Dict[str, Any]:
        action_data = request.action
        action = CloudFinOpsAction(**action_data)
        obs = _env.step(action, timeout_s=request.timeout_s)
        obs_dict = obs.model_dump()
        return {
            "observation": obs_dict,
            "reward": obs.reward,
            "done": obs.done,
        }

    @app.get("/state")
    async def state_endpoint() -> Dict[str, Any]:
        return _env.state.model_dump()


# ---------------------------------------------------------------------------
# Extra endpoints (always registered regardless of create_app usage)
# ---------------------------------------------------------------------------

@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/tasks")
async def list_tasks() -> Dict[str, Any]:
    from server.tasks import TASKS
    return {
        "tasks": [
            {
                "id": t.task_id,
                "name": t.name,
                "difficulty": t.difficulty,
                "max_steps": t.max_steps,
                "description": t.description,
            }
            for t in TASKS.values()
        ]
    }
