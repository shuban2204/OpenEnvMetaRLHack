---
title: Cloud FinOps Environment
emoji: "\U00002601"
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
tags:
  - openenv
---

# Cloud FinOps & Infrastructure Right-Sizing Environment

An **OpenEnv** environment where an AI agent acts as a **Cloud Financial Engineer**, optimizing cloud infrastructure spend by deleting unused storage, terminating idle instances, and right-sizing over-provisioned resources — all while respecting SLA constraints and application dependencies.

## Motivation

Cloud cost optimization is a **$30B+ real-world problem**. Organizations routinely overspend 30–40% on cloud bills due to:
- Unattached storage volumes left behind after migrations
- Idle instances running forgotten staging/dev workloads
- Over-provisioned instances using m5.2xlarge when m5.large would suffice

This environment simulates a realistic AWS-like infrastructure fleet with CPU/memory utilization metrics, application dependency graphs, and pricing data — letting RL agents learn cost-optimization strategies.

## Environment Overview

| Property | Value |
|---|---|
| **Action Space** | `terminate_instance(id)`, `delete_volume(id)`, `resize_instance(id, new_type)`, `skip`, `submit` |
| **Observation Space** | Instance list (type, CPU/mem utilization, cost, dependencies), Volume list (size, state, attachment), spend totals, violations |
| **Reward** | Per-step: proportional to $ saved / optimal $ savings. Penalties for SLA violations. Final graded score 0.0–1.0 |
| **Episode Length** | 10–20 steps depending on task |

## Tasks

### 1. Cleanup Unused Volumes (Easy)
- **Objective**: Delete all unattached EBS volumes
- **Infrastructure**: 6 instances, 10 volumes (4 unattached)
- **Max Steps**: 10
- **Grading**: % of unattached volumes correctly deleted. Penalty for deleting attached volumes.
- **Optimal Savings**: ~$122.75/month

### 2. Right-Size Over-Provisioned Instances (Medium)
- **Objective**: Resize under-utilized instances to cheaper types without SLA violations
- **Infrastructure**: 8 instances (5 over-provisioned), 4 volumes
- **Max Steps**: 12
- **Grading**: Cost savings achieved vs. optimal savings. Penalty for causing SLA violations (peak CPU > 80% of new capacity).
- **Optimal Savings**: ~$574.28/month

### 3. Full Cost Optimization (Hard)
- **Objective**: Comprehensive fleet optimization — volumes + terminations + right-sizing + dependency awareness
- **Infrastructure**: 15 instances, 12 volumes, app dependency graph, 7-day CPU history
- **Max Steps**: 20
- **Grading**: Composite — 50% savings ratio + 25% zero violations + 25% action completeness
- **Optimal Savings**: ~$1,004/month

## Observation Details

Each observation includes:
```json
{
  "instances": [
    {
      "instance_id": "i-201",
      "instance_type": "m5.2xlarge",
      "state": "running",
      "app_name": "prod-api",
      "cpu_avg_percent": 55.0,
      "cpu_peak_percent": 82.0,
      "memory_avg_percent": 60.0,
      "monthly_cost": 276.48,
      "cpu_history_7d": [52, 54, 55, 58, 55, 53, 56],
      "dependencies": ["staging-api"]
    }
  ],
  "volumes": [
    {
      "volume_id": "vol-202",
      "size_gb": 200,
      "volume_type": "gp3",
      "state": "available",
      "attached_instance_id": null,
      "monthly_cost": 16.0
    }
  ],
  "current_monthly_spend": 2438.87,
  "savings_achieved": 0.0,
  "violations": [],
  "resize_options": {"m5.2xlarge": ["m5.xlarge", "m5.large"], ...}
}
```

## Action Details

```json
{"action_type": "delete_volume", "target_id": "vol-202"}
{"action_type": "terminate_instance", "target_id": "i-202"}
{"action_type": "resize_instance", "target_id": "i-203", "new_type": "c5.large"}
{"action_type": "skip"}
{"action_type": "submit"}
```

## SLA Violation Model

When right-sizing, the environment checks whether the workload would fit on the new instance type:

```
effective_peak_cpu = peak_cpu × (old_capacity / new_capacity)
```

If `effective_peak_cpu > 80%`, the resize causes an **SLA violation** (application may crash under peak load). Instance capacities within a family scale as:

| Type | Capacity |
|------|----------|
| `*.2xlarge` | 4x |
| `*.xlarge` | 2x |
| `*.large` | 1x |
| `t3.medium` | 2x |
| `t3.small` | 1x |
| `t3.micro` | 0.5x |

## Setup

### Prerequisites
- Python 3.10+
- Docker (for containerized deployment)

### Local Development
```bash
pip install -r requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

### Docker
```bash
docker build -t cloud-finops-env .
docker run -p 7860:7860 cloud-finops-env
```

### Run Inference
```bash
export HF_TOKEN=your_api_key
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
python inference.py
```

## Baseline Scores

| Task | Difficulty | Baseline Score |
|------|-----------|---------------|
| cleanup_unused_volumes | Easy | ~0.85–1.0 |
| rightsize_overprovisioned | Medium | ~0.50–0.75 |
| full_cost_optimization | Hard | ~0.30–0.55 |

## Project Structure

```
├── openenv.yaml                 # OpenEnv manifest
├── pyproject.toml               # Python project config
├── requirements.txt             # Dependencies
├── Dockerfile                   # Container build
├── inference.py                 # Baseline LLM agent
├── README.md
├── cloud_finops_env/
│   ├── __init__.py              # Package exports
│   ├── models.py                # Action, Observation, State models
│   └── client.py                # EnvClient subclass
└── server/
    ├── __init__.py
    ├── app.py                   # FastAPI server
    ├── environment.py           # Core environment logic
    ├── simulator.py             # Cloud infrastructure simulation
    └── tasks.py                 # Task definitions & graders
```
