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

An **OpenEnv** environment where an AI agent acts as a **Cloud Financial Engineer**, optimizing cloud infrastructure spend by deleting unused storage, terminating idle instances, right-sizing over-provisioned resources, migrating to spot pricing, and planning reserved instance purchases — all while respecting SLA constraints, application dependencies, and pricing models.

## Motivation

Cloud cost optimization is a **$30B+ real-world problem**. Organizations routinely overspend 30-40% on cloud bills due to:
- Unattached storage volumes left behind after migrations
- Idle instances running forgotten staging/dev workloads
- Over-provisioned instances using m5.2xlarge when m5.large would suffice
- On-demand pricing for workloads that should be on spot or reserved instances

This environment simulates a realistic AWS-like infrastructure fleet with CPU/memory utilization metrics, 7-day history, application dependency graphs, instance tags (team, env, tier), pricing models, and real AWS pricing data.

## Environment Overview

| Property | Value |
|---|---|
| **Action Space** | `terminate_instance`, `delete_volume`, `resize_instance`, `convert_to_spot`, `purchase_ri`, `skip`, `submit` |
| **Observation Space** | Instances (type, CPU/mem, cost, tags, region, dependencies, pricing_model, uptime), Volumes (size, state, dates), spot/RI pricing tables |
| **Reward** | Per-step: proportional to $/savings normalized by optimal. Penalties for SLA violations. Final graded score 0.0-1.0 |
| **Episode Length** | 10-20 steps depending on task |

## Tasks (5 tasks, easy -> expert)

### 1. Cleanup Unused Volumes (Easy)
- **Objective**: Delete all unattached EBS volumes (state='available')
- **Infrastructure**: 6 instances, 10 volumes (4 unattached)
- **Max Steps**: 10
- **Grading**: Savings ratio. Penalty for deleting attached volumes.
- **Optimal Savings**: ~$122.75/month

### 2. Right-Size Over-Provisioned Instances (Medium)
- **Objective**: Resize under-utilized instances without SLA violations
- **Infrastructure**: 8 instances (5 over-provisioned), 4 volumes
- **Max Steps**: 12
- **Grading**: Savings ratio with violation penalty. Peak CPU must stay < 80% of new capacity.
- **Optimal Savings**: ~$574.28/month

### 3. Spot Instance Migration (Medium-Hard)
- **Objective**: Convert fault-tolerant workloads to spot pricing (60-70% savings)
- **Infrastructure**: 10 instances with tags (stateful, tier, env), some with dependencies
- **Max Steps**: 15
- **Traps**: Instance with dependencies that looks like a batch worker; dev instance that's actually stateful (local DB)
- **Grading**: 60% savings + 40% zero-violations
- **Optimal Savings**: ~$449/month

### 4. Full Cost Optimization (Hard)
- **Objective**: Comprehensive fleet optimization across 16 instances and 12 volumes
- **Infrastructure**: App dependency graph, 7-day CPU history per instance, multi-region deployment, enriched tags
- **Max Steps**: 20
- **Traps**: Weekend-spike instance (low avg=18% but peak=72% in 7d history); critical instances with dependencies
- **Grading**: 50% savings + 25% zero-violations + 25% action completeness
- **Optimal Savings**: ~$1,004/month

### 5. Reserved Instance Planning (Expert)
- **Objective**: Commit to 1-year RIs for stable, long-running workloads (30-40% savings)
- **Infrastructure**: 12 instances with varying CPU stability, uptime, deprecation status
- **Max Steps**: 18
- **Traps**: Deprecated instance with stable CPU (about to be decommissioned); nightly-build with wild CPU spikes; low-uptime new instances
- **Grading**: 40% savings + 30% zero-violations + 30% precision (correct RI purchases / total RI purchases)
- **Optimal Savings**: ~$357/month

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
      "dependencies": ["staging-api"],
      "tags": {"team": "backend", "env": "prod", "tier": "critical"},
      "region": "us-east-1",
      "launch_date": "2025-06-15",
      "pricing_model": "on-demand",
      "network_io_gbps": 4.5,
      "uptime_days": 365
    }
  ],
  "volumes": [...],
  "current_monthly_spend": 2540.29,
  "savings_achieved": 0.0,
  "violations": [],
  "resize_options": {"m5.2xlarge": ["m5.xlarge", "m5.large"]},
  "spot_pricing": {"m5.2xlarge": 82.94, "c5.xlarge": 36.72},
  "ri_pricing": {"m5.2xlarge": 179.71, "c5.xlarge": 79.56}
}
```

## Action Space

```json
{"action_type": "delete_volume", "target_id": "vol-202"}
{"action_type": "terminate_instance", "target_id": "i-202"}
{"action_type": "resize_instance", "target_id": "i-203", "new_type": "c5.large"}
{"action_type": "convert_to_spot", "target_id": "i-301"}
{"action_type": "purchase_ri", "target_id": "i-401"}
{"action_type": "skip"}
{"action_type": "submit"}
```

## SLA Violation Model

```
effective_peak_cpu = peak_cpu x (old_capacity / new_capacity)
```

If `effective_peak_cpu > 80%`, the resize causes an **SLA violation**. Capacities:

| Type | Capacity |
|------|----------|
| `*.2xlarge` | 4x |
| `*.xlarge` | 2x |
| `*.large` | 1x |
| `t3.medium` | 2x |
| `t3.small` | 1x |
| `t3.micro` | 0.5x |

## Setup

### Local Development
```bash
pip install -r requirements.txt
python -m server.app
```

### Docker
```bash
docker build -t cloud-finops-env .
docker run -p 7860:7860 cloud-finops-env
```

### Run Tests
```bash
pip install pytest
PYTHONPATH=. pytest tests/ -v
```

### Run Inference
```bash
export HF_TOKEN=your_api_key
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
python inference.py
```

## Baseline Scores

| Task | Difficulty | Baseline Score | Steps | Model |
|------|-----------|---------------|-------|-------|
| cleanup_unused_volumes | Easy | **1.000** | 5 | Qwen2.5-72B-Instruct |
| rightsize_overprovisioned | Medium | **0.400** | 12 | Qwen2.5-72B-Instruct |
| spot_instance_migration | Medium-Hard | **1.000** | 14 | Qwen2.5-72B-Instruct |
| full_cost_optimization | Hard | **0.644** | 20 | Qwen2.5-72B-Instruct |
| reserved_instance_planning | Expert | **1.000** | 7 | Qwen2.5-72B-Instruct |

## Architecture

Agent → FastAPI Server → Environment Engine → Simulator → State & Reward → Agent

+----------------------+
|   Agent (LLM / RL)   |
+----------------------+
           |
           v
+----------------------+
|  FastAPI Server      |
|     (app.py)         |
+----------------------+
           |
           v
+----------------------+
|  Environment Engine  |
|  (environment.py)    |
+----------------------+
           |
           v
+----------------------+
|     Simulator        |
|   (simulator.py)     |
+----------------------+
           |
           v
+----------------------+
|   State + Reward     |
| (Returned to Agent)  |
+----------------------+

## Project Structure

```
├── openenv.yaml                 # OpenEnv manifest (5 tasks)
├── pyproject.toml               # Python project config
├── requirements.txt             # Dependencies
├── Dockerfile                   # Container build
├── inference.py                 # Baseline LLM agent
├── README.md
├── cloud_finops_env/
│   ├── __init__.py              # Package exports
│   ├── models.py                # Action, Observation, State models
│   └── client.py                # EnvClient subclass
├── server/
│   ├── __init__.py
│   ├── app.py                   # FastAPI server
│   ├── environment.py           # Core environment logic
│   ├── simulator.py             # Cloud infrastructure simulation
│   └── tasks.py                 # Task definitions & graders
└── tests/
    └── test_environment.py      # 32 pytest tests
```
