"""Deterministic in-process smoke environment for harness verification."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

from .models import AgentDecision, TaskSpec, Transition


class SmokeAgent:
    name = "v0.1-smoke-agent"

    def act(self, observation, legal_actions, context) -> AgentDecision:
        return AgentDecision(
            action={"type": "gather", "entity_ids": [1], "target_entity_id": 2},
            prompt_tokens=12,
            completion_tokens=4,
            latency_ms=2.0,
            rationale="increase the target wood stockpile",
        )


class EconomySmokeEnvironment:
    """Small deterministic environment that exercises the benchmark pipeline."""

    def __init__(self):
        self.step_number = 0
        self.wood = 300

    def _observation(self) -> dict:
        player = {
            "id": 1,
            "state": "active",
            "phase": "village",
            "resources": {"food": 300, "wood": self.wood, "stone": 300, "metal": 300},
            "resource_gatherers": {"food": 0, "wood": 6, "stone": 0, "metal": 0},
            "population": {"used": 7, "limit": 20},
            "units": {"support_civilian": 3, "infantry_spearman_b": 4},
            "unit_count": 7,
            "structures": {"civil_centre": 1},
            "structure_count": 1,
            "statistics": {"worker_count": 7},
        }
        opponent = dict(player)
        opponent.update(id=2, resources=dict(player["resources"]))
        return {
            "schema_version": "0.1",
            "episode_id": "smoke-econ-001",
            "step": self.step_number,
            "game_time_ms": self.step_number * 200,
            "track": "symbolic",
            "player": player,
            "opponent": opponent,
            "map": {},
            "entities": [],
            "events": {},
            "event_log": [],
        }

    def reset(self, task: TaskSpec, seed: int) -> Mapping:
        self.step_number = 0
        self.wood = 300
        return self._observation()

    def legal_actions(self) -> Sequence[str]:
        return ("wait", "gather")

    def validate_action(self, action: Mapping) -> bool:
        return action.get("type") in self.legal_actions()

    def step(self, action: Mapping) -> Transition:
        self.step_number += 1
        if action.get("type") == "gather":
            self.wood += 50
        return Transition(
            self._observation(),
            reward=min(1.0, self.wood / 500),
            info={"economy_reward": min(1.0, self.wood / 500), "survival_reward": 1.0},
        )

    def apply_perturbation(self, perturbation: Mapping) -> None:
        return None

    def export_replay(self, destination: Path):
        return None

    def close(self) -> None:
        return None
