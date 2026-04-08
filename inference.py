"""
Inference Script — Cloud FinOps Environment
=============================================
Baseline agent that uses an OpenAI-compatible LLM to optimize cloud costs.

MANDATORY ENV VARS:
    API_BASE_URL    The LLM endpoint       (default: https://router.huggingface.co/v1)
    MODEL_NAME      The model identifier   (default: Qwen/Qwen2.5-72B-Instruct)
    HF_TOKEN        Your API key

STDOUT FORMAT:
    [START] task=<task> env=<env> model=<model>
    [STEP]  step=<n> action=<action> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> rewards=<r1,r2,...>
"""

from __future__ import annotations

import json
import os
import textwrap
from typing import Any, Dict, List, Optional

import requests
from openai import OpenAI

# Pricing reference for prompt building
INSTANCE_PRICING = {
    "t3.micro": 7.49, "t3.small": 15.04, "t3.medium": 30.09, "t3.large": 60.12,
    "m5.large": 69.12, "m5.xlarge": 138.24, "m5.2xlarge": 276.48,
    "c5.large": 61.20, "c5.xlarge": 122.40, "c5.2xlarge": 244.80,
    "r5.large": 90.72, "r5.xlarge": 181.44,
}

# ── Configuration ───────────────────────────────────────────────────────────

API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
ENV_URL = os.getenv("ENV_URL", "http://localhost:7860")
BENCHMARK = "cloud_finops_env"
TEMPERATURE = 0.2
MAX_TOKENS = 512

TASK_IDS = [
    "cleanup_unused_volumes",
    "rightsize_overprovisioned",
    "spot_instance_migration",
    "full_cost_optimization",
    "reserved_instance_planning",
]

TASK_MAX_STEPS = {
    "cleanup_unused_volumes": 10,
    "rightsize_overprovisioned": 12,
    "spot_instance_migration": 15,
    "full_cost_optimization": 20,
    "reserved_instance_planning": 18,
}


# ── Logging (mandatory stdout format) ──────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={error or 'null'}",
        flush=True,
    )


def log_end(success: bool, steps: int, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}",
        flush=True,
    )


# ── Environment HTTP helpers ────────────────────────────────────────────────

def env_reset(task_id: str) -> Dict[str, Any]:
    resp = requests.post(
        f"{ENV_URL}/reset",
        json={"task_id": task_id},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def env_step(action: Dict[str, Any]) -> Dict[str, Any]:
    resp = requests.post(
        f"{ENV_URL}/step",
        json={"action": action},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


# ── System prompt ───────────────────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""\
You are a Cloud Financial Engineer AI agent. Your job is to analyze cloud
infrastructure and take actions to reduce monthly spend while keeping
applications running safely.

You will receive observations containing:
- instances: list of EC2 instances with CPU/memory utilization, tags, region, pricing_model
- volumes: list of EBS volumes with attachment status and dates
- current_monthly_spend, savings_achieved, violations
- task_description: what you should do
- resize_options: valid downgrade paths for each instance type
- spot_pricing: monthly cost if converted to spot
- ri_pricing: monthly cost if 1-year reserved instance purchased
- action_history: what you've already done

You must respond with EXACTLY ONE JSON action per turn:

{"action_type": "delete_volume", "target_id": "vol-XXX"}
{"action_type": "terminate_instance", "target_id": "i-XXX"}
{"action_type": "resize_instance", "target_id": "i-XXX", "new_type": "m5.large"}
{"action_type": "convert_to_spot", "target_id": "i-XXX"}
{"action_type": "purchase_ri", "target_id": "i-XXX"}
{"action_type": "skip"}
{"action_type": "submit"}

RULES:
- Only delete volumes with state="available" (unattached).
- Only terminate instances with very low CPU (<5%) AND empty dependencies list.
- Only resize to types listed in resize_options for that instance type.
- After resizing, effective peak CPU = peak_cpu * (old_capacity / new_capacity)
  must stay BELOW 80%. Capacity roughly halves with each downgrade step.
- convert_to_spot: ONLY for stateless (tag stateful!="true"), non-critical
  (tag tier!="critical"), no-dependency instances. Spot = 60-70% savings.
- purchase_ri: ONLY for long-running (uptime>=180d), stable CPU (low 7d variance),
  production instances NOT tagged deprecated/migrating. RI = 30-40% savings.
- Use "submit" when you have completed all optimizations.
- Respond with ONLY the JSON object — no explanation, no markdown.
""")


# ── Build user prompt from observation ──────────────────────────────────────

def build_user_prompt(obs: Dict[str, Any], step: int) -> str:
    obs_data = obs.get("observation", obs)
    parts = [
        f"STEP {step}/{obs_data.get('max_steps', '?')}",
        f"Task: {obs_data.get('task_description', 'N/A')}\n",
        f"Current monthly spend: ${obs_data.get('current_monthly_spend', 0):.2f}",
        f"Savings achieved so far: ${obs_data.get('savings_achieved', 0):.2f}",
    ]

    violations = obs_data.get("violations", [])
    if violations:
        parts.append(f"Violations: {violations}")

    parts.append(f"\nActions taken so far: {obs_data.get('action_history', [])}")

    # Instances
    instances = obs_data.get("instances", [])
    parts.append(f"\n--- INSTANCES ({len(instances)}) ---")
    for inst in instances:
        line = (
            f"  {inst['instance_id']} | {inst['instance_type']} | "
            f"{inst['app_name']} | CPU avg={inst['cpu_avg_percent']}% "
            f"peak={inst['cpu_peak_percent']}% | mem={inst['memory_avg_percent']}% "
            f"| ${inst['monthly_cost']:.2f}/mo"
        )
        deps = inst.get("dependencies", [])
        if deps:
            line += f" | depended_on_by={deps}"
        cpu_hist = inst.get("cpu_history_7d", [])
        if cpu_hist:
            line += f" | 7d_cpu={cpu_hist}"
        tags = inst.get("tags", {})
        if tags:
            line += f" | tags={tags}"
        pricing = inst.get("pricing_model", "")
        if pricing and pricing != "on-demand":
            line += f" | pricing={pricing}"
        uptime = inst.get("uptime_days", 0)
        if uptime:
            line += f" | uptime={uptime}d"
        parts.append(line)

    # Volumes
    volumes = obs_data.get("volumes", [])
    parts.append(f"\n--- VOLUMES ({len(volumes)}) ---")
    for vol in volumes:
        attached = vol.get("attached_instance_id") or "NONE (unattached)"
        parts.append(
            f"  {vol['volume_id']} | {vol['size_gb']}GB {vol['volume_type']} | "
            f"state={vol['state']} | attached_to={attached} | "
            f"${vol['monthly_cost']:.2f}/mo"
        )

    # Resize options
    resize_opts = obs_data.get("resize_options", {})
    if resize_opts:
        parts.append("\n--- RESIZE OPTIONS ---")
        for itype, targets in resize_opts.items():
            parts.append(f"  {itype} -> {targets}")

    # Spot pricing
    spot_prices = obs_data.get("spot_pricing", {})
    if spot_prices:
        parts.append("\n--- SPOT PRICING (monthly) ---")
        for itype, price in spot_prices.items():
            parts.append(f"  {itype}: ${price:.2f}/mo (on-demand: ${INSTANCE_PRICING.get(itype, 0):.2f})")

    # RI pricing
    ri_prices = obs_data.get("ri_pricing", {})
    if ri_prices:
        parts.append("\n--- RESERVED INSTANCE PRICING (1yr, monthly) ---")
        for itype, price in ri_prices.items():
            parts.append(f"  {itype}: ${price:.2f}/mo (on-demand: ${INSTANCE_PRICING.get(itype, 0):.2f})")

    parts.append("\nRespond with ONE JSON action:")
    return "\n".join(parts)


# ── LLM call ────────────────────────────────────────────────────────────────

def get_llm_action(client: OpenAI, obs: Dict[str, Any], step: int) -> Dict[str, Any]:
    user_prompt = build_user_prompt(obs, step)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        # Strip markdown code fences if present
        if text.startswith("```"):
            text = text.split("\n", 1)[-1]
            if text.endswith("```"):
                text = text[:-3].strip()
            elif "```" in text:
                text = text[: text.rfind("```")].strip()
        action = json.loads(text)
        if "action_type" not in action:
            action = {"action_type": "skip"}
        return action
    except Exception as exc:
        print(f"[DEBUG] LLM call failed: {exc}", flush=True)
        return {"action_type": "skip"}


# ── Run one episode ─────────────────────────────────────────────────────────

def run_episode(client: OpenAI, task_id: str) -> None:
    max_steps = TASK_MAX_STEPS.get(task_id, 15)
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        obs = env_reset(task_id)

        for step in range(1, max_steps + 1):
            if obs.get("done", False):
                break

            action = get_llm_action(client, obs, step)
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
                error=None,
            )

            if done:
                break

        # Final score = last reward on done (the graded score)
        if rewards:
            score = rewards[-1] if obs.get("done", False) else sum(rewards)
        score = max(0.0, min(score, 1.0))
        success = score >= 0.1

    except Exception as exc:
        print(f"[DEBUG] Episode error: {exc}", flush=True)

    log_end(success=success, steps=steps_taken, rewards=rewards)


# ── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    for task_id in TASK_IDS:
        print(f"\n{'='*60}", flush=True)
        print(f"  Running task: {task_id}", flush=True)
        print(f"{'='*60}\n", flush=True)
        run_episode(client, task_id)


if __name__ == "__main__":
    main()
