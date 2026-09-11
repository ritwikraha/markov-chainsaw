import json

from zero_ad_bench import AgentDecision, BenchmarkRunner, TaskSpec, Transition


class IncrementAgent:
    name = "increment-agent"

    def act(self, observation, legal_actions, context):
        return AgentDecision(
            action={"type": "gather", "entity_ids": [1], "target_entity_id": 2},
            prompt_tokens=10,
            completion_tokens=2,
            latency_ms=5,
            rationale="increase wood",
        )


class FakeEnvironment:
    def __init__(self):
        self.step_number = 0
        self.wood = 0
        self.closed = False

    def observation(self):
        return {
            "schema_version": "0.1",
            "episode_id": "fake",
            "step": self.step_number,
            "game_time_ms": self.step_number * 200,
            "track": "symbolic",
            "player": {"resources": {"wood": self.wood}, "state": "active"},
            "opponent": {"state": "active"},
            "entities": [],
            "events": {},
        }

    def reset(self, task, seed):
        return self.observation()

    def legal_actions(self):
        return ["wait", "gather"]

    def validate_action(self, action):
        return action.get("type") in self.legal_actions()

    def step(self, action):
        self.step_number += 1
        if action["type"] == "gather":
            self.wood += 100
        return Transition(
            self.observation(),
            info={"economy_reward": min(1, self.wood / 200), "survival_reward": 1},
        )

    def apply_perturbation(self, perturbation):
        self.wood = 0

    def export_replay(self, destination):
        return None

    def close(self):
        self.closed = True


def test_runner_writes_complete_episode(tmp_path):
    task = TaskSpec(
        id="econ-999",
        version="0.1",
        split="development",
        category="economy",
        title="Runner check",
        description="Exercise the complete episode writer.",
        tracks=("symbolic",),
        scenario={"seed": 9},
        budget={"max_steps": 3, "decision_interval": 1, "max_agent_actions": 3, "max_tokens": 100, "max_mean_latency_ms": 100},
        goal={"aggregation": "all", "predicates": [{"path": "player.resources.wood", "op": "gte", "value": 200}]},
    )
    environment = FakeEnvironment()
    summary = BenchmarkRunner(tmp_path).run(task, IncrementAgent(), environment, episode_id="episode-test")
    assert summary["success"] is True
    assert summary["metrics"]["completed_steps"] == 2
    assert environment.closed is True
    episode = tmp_path / "episode-test"
    assert len(list((episode / "observations").glob("*.json"))) == 3
    assert len(list((episode / "evaluator_observations").glob("*.json"))) == 3
    assert len((episode / "actions.jsonl").read_text().splitlines()) == 2
    assert json.loads((episode / "summary.json").read_text())["score"]["total"] == summary["score"]["total"]
