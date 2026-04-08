"""Tests for Cloud FinOps Environment."""

import pytest
from server.environment import CloudFinOpsEnvironment
from cloud_finops_env.models import CloudFinOpsAction


@pytest.fixture
def env():
    return CloudFinOpsEnvironment()


# ── Reset tests ─────────────────────────────────────────────────────────────

class TestReset:
    def test_reset_produces_clean_state(self, env):
        obs = env.reset(task_id="cleanup_unused_volumes")
        assert obs.done is False
        assert obs.reward is None
        assert obs.step_number == 0
        assert obs.savings_achieved == 0.0
        assert obs.violations == []
        assert obs.action_history == []
        assert len(obs.instances) > 0
        assert len(obs.volumes) > 0

    def test_reset_all_tasks(self, env):
        task_ids = [
            "cleanup_unused_volumes", "rightsize_overprovisioned",
            "spot_instance_migration", "full_cost_optimization",
            "reserved_instance_planning",
        ]
        for task_id in task_ids:
            obs = env.reset(task_id=task_id)
            assert obs.done is False
            assert obs.task_id == task_id
            assert len(obs.instances) >= 6

    def test_reset_clears_previous_state(self, env):
        obs = env.reset(task_id="cleanup_unused_volumes")
        env.step(CloudFinOpsAction(action_type="delete_volume", target_id="vol-003"))
        obs = env.reset(task_id="cleanup_unused_volumes")
        assert obs.savings_achieved == 0.0
        assert obs.step_number == 0

    def test_invalid_task_falls_back(self, env):
        obs = env.reset(task_id="nonexistent_task")
        assert obs.task_id == "cleanup_unused_volumes"


# ── Grader score range tests ───────────────────────────────────────────────

class TestGraderScoreRange:
    @pytest.mark.parametrize("task_id", [
        "cleanup_unused_volumes", "rightsize_overprovisioned",
        "spot_instance_migration", "full_cost_optimization",
        "reserved_instance_planning",
    ])
    def test_no_action_score_in_range(self, env, task_id):
        env.reset(task_id=task_id)
        obs = env.step(CloudFinOpsAction(action_type="submit"))
        assert 0.0 <= obs.reward <= 1.0

    @pytest.mark.parametrize("task_id", [
        "cleanup_unused_volumes", "rightsize_overprovisioned",
        "spot_instance_migration", "full_cost_optimization",
        "reserved_instance_planning",
    ])
    def test_score_deterministic(self, env, task_id):
        env.reset(task_id=task_id)
        obs1 = env.step(CloudFinOpsAction(action_type="submit"))
        env.reset(task_id=task_id)
        obs2 = env.step(CloudFinOpsAction(action_type="submit"))
        assert obs1.reward == obs2.reward


# ── Easy task tests ─────────────────────────────────────────────────────────

class TestEasyTask:
    def test_perfect_score(self, env):
        env.reset(task_id="cleanup_unused_volumes")
        for vid in ["vol-003", "vol-006", "vol-008", "vol-010"]:
            env.step(CloudFinOpsAction(action_type="delete_volume", target_id=vid))
        obs = env.step(CloudFinOpsAction(action_type="submit"))
        assert obs.reward == 1.0

    def test_deleting_attached_volume_penalized(self, env):
        env.reset(task_id="cleanup_unused_volumes")
        obs = env.step(CloudFinOpsAction(action_type="delete_volume", target_id="vol-001"))
        assert obs.reward < 0
        assert len(obs.violations) == 1

    def test_partial_score(self, env):
        env.reset(task_id="cleanup_unused_volumes")
        env.step(CloudFinOpsAction(action_type="delete_volume", target_id="vol-003"))
        obs = env.step(CloudFinOpsAction(action_type="submit"))
        assert 0.0 < obs.reward < 1.0


# ── Medium task tests ──────────────────────────────────────────────────────

class TestMediumTask:
    def test_perfect_score(self, env):
        env.reset(task_id="rightsize_overprovisioned")
        resizes = [
            ("i-101", "m5.large"), ("i-102", "c5.large"),
            ("i-104", "m5.large"), ("i-105", "t3.small"),
            ("i-107", "m5.large"),
        ]
        for iid, nt in resizes:
            env.step(CloudFinOpsAction(action_type="resize_instance", target_id=iid, new_type=nt))
        obs = env.step(CloudFinOpsAction(action_type="submit"))
        assert obs.reward == 1.0

    def test_sla_violation_penalized(self, env):
        env.reset(task_id="rightsize_overprovisioned")
        obs = env.step(CloudFinOpsAction(action_type="resize_instance", target_id="i-103", new_type="r5.large"))
        assert obs.reward < 0
        assert "VIOLATION" in obs.violations[0]


# ── Spot migration tests ──────────────────────────────────────────────────

class TestSpotTask:
    def test_perfect_score(self, env):
        env.reset(task_id="spot_instance_migration")
        for iid in ["i-301", "i-302", "i-303", "i-304", "i-310"]:
            env.step(CloudFinOpsAction(action_type="convert_to_spot", target_id=iid))
        obs = env.step(CloudFinOpsAction(action_type="submit"))
        assert obs.reward == 1.0

    def test_stateful_conversion_penalized(self, env):
        env.reset(task_id="spot_instance_migration")
        obs = env.step(CloudFinOpsAction(action_type="convert_to_spot", target_id="i-307"))
        assert obs.reward < 0
        assert "stateful" in obs.last_action_error.lower()

    def test_dependency_conversion_penalized(self, env):
        env.reset(task_id="spot_instance_migration")
        obs = env.step(CloudFinOpsAction(action_type="convert_to_spot", target_id="i-308"))
        assert obs.reward < 0
        assert "depend" in obs.last_action_error.lower()


# ── Hard task tests ────────────────────────────────────────────────────────

class TestHardTask:
    def test_perfect_score(self, env):
        env.reset(task_id="full_cost_optimization")
        for vid in ["vol-202", "vol-204", "vol-207", "vol-209", "vol-212"]:
            env.step(CloudFinOpsAction(action_type="delete_volume", target_id=vid))
        for iid in ["i-202", "i-204", "i-209", "i-210"]:
            env.step(CloudFinOpsAction(action_type="terminate_instance", target_id=iid))
        for iid, nt in [("i-203", "c5.large"), ("i-206", "m5.large"), ("i-211", "c5.large"), ("i-215", "m5.large")]:
            env.step(CloudFinOpsAction(action_type="resize_instance", target_id=iid, new_type=nt))
        obs = env.step(CloudFinOpsAction(action_type="submit"))
        assert obs.reward == 1.0

    def test_dependency_termination_blocked(self, env):
        env.reset(task_id="full_cost_optimization")
        obs = env.step(CloudFinOpsAction(action_type="terminate_instance", target_id="i-205"))
        assert obs.reward < 0
        assert any("i-205" in v for v in obs.violations)

    def test_weekend_spike_trap(self, env):
        env.reset(task_id="full_cost_optimization")
        obs = env.step(CloudFinOpsAction(action_type="resize_instance", target_id="i-216", new_type="m5.large"))
        assert obs.reward < 0
        assert "VIOLATION" in obs.last_action_error


# ── RI planning tests ──────────────────────────────────────────────────────

class TestRITask:
    def test_perfect_score(self, env):
        env.reset(task_id="reserved_instance_planning")
        for iid in ["i-401", "i-402", "i-403", "i-404", "i-409", "i-412"]:
            env.step(CloudFinOpsAction(action_type="purchase_ri", target_id=iid))
        obs = env.step(CloudFinOpsAction(action_type="submit"))
        assert obs.reward == 1.0

    def test_deprecated_instance_penalized(self, env):
        env.reset(task_id="reserved_instance_planning")
        obs = env.step(CloudFinOpsAction(action_type="purchase_ri", target_id="i-408"))
        assert obs.reward < 0
        assert "decommission" in obs.last_action_error.lower()

    def test_variable_cpu_penalized(self, env):
        env.reset(task_id="reserved_instance_planning")
        obs = env.step(CloudFinOpsAction(action_type="purchase_ri", target_id="i-410"))
        assert obs.reward < 0
        assert "variable" in obs.last_action_error.lower()

    def test_new_instance_penalized(self, env):
        env.reset(task_id="reserved_instance_planning")
        obs = env.step(CloudFinOpsAction(action_type="purchase_ri", target_id="i-411"))
        assert obs.reward < 0
        assert "uptime" in obs.last_action_error.lower()


# ── API endpoint tests ─────────────────────────────────────────────────────

class TestAPI:
    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from server.app import app
        return TestClient(app)

    def test_health(self, client):
        assert client.get("/health").status_code == 200

    def test_tasks_endpoint(self, client):
        r = client.get("/tasks")
        assert r.status_code == 200
        assert len(r.json()["tasks"]) == 5

    def test_reset_step_state_cycle(self, client):
        r = client.post("/reset", json={"task_id": "cleanup_unused_volumes"})
        assert r.status_code == 200
        assert r.json()["done"] is False

        r = client.post("/step", json={"action": {"action_type": "skip"}})
        assert r.status_code == 200

        r = client.get("/state")
        assert r.status_code == 200
        assert "episode_id" in r.json()
