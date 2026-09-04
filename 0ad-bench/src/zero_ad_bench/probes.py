"""Deterministic symbolic probes for checking model and harness compatibility.

These probes exercise observation parsing, RTS decisions, macro-action JSON, and
trajectory recording without claiming engine-backed 0 A.D. task results.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

from .models import TaskSpec, Transition


@dataclass(frozen=True)
class ProbeCase:
    id: str
    category: str
    objective: str
    state: Mapping[str, Any]
    candidates: Sequence[Mapping[str, Any]]
    expected_action: Mapping[str, Any]

    def task(self) -> TaskSpec:
        return TaskSpec(
            id=self.id,
            version="0.1-model-compatibility",
            split="development",
            category=self.category,
            title=self.objective,
            description=self.objective,
            tracks=("symbolic",),
            scenario={"fixture": "in-process-model-probe", "seed": 4100},
            budget={
                "max_steps": 1,
                "decision_interval": 1,
                "max_agent_actions": 1,
                "max_tokens": 512,
                "max_mean_latency_ms": 60_000,
            },
            goal={
                "aggregation": "all",
                "terminate_on_success": True,
                "predicates": [
                    {"path": "events.probe_success", "op": "eq", "value": True}
                ],
            },
            tags=("model-compatibility", "symbolic", "single-decision"),
        )


def _case(
    id: str,
    category: str,
    objective: str,
    state: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    expected: int,
) -> ProbeCase:
    return ProbeCase(id, category, objective, state, tuple(candidates), dict(candidates[expected]))


def default_probe_cases() -> Tuple[ProbeCase, ...]:
    """Return the versioned 15-probe development suite."""
    return (
        _case(
            "probe-econ-001", "economy",
            "Prevent a wood shortage. Assign the three idle workers to the nearby wood line.",
            {"resources": {"food": 620, "wood": 45, "stone": 210, "metal": 180}, "idle_workers": [1, 2, 3], "resource_nodes": {"101": "food", "102": "wood"}},
            [
                {"type": "gather", "entity_ids": [1, 2, 3], "target_entity_id": 101},
                {"type": "gather", "entity_ids": [1, 2, 3], "target_entity_id": 102},
                {"type": "wait"},
            ], 1,
        ),
        _case(
            "probe-econ-002", "economy",
            "Grow the economy by training one worker from the civic centre while population space is available.",
            {"resources": {"food": 300, "wood": 250}, "population": {"used": 12, "limit": 20}, "structures": {"civic_centre": [10]}},
            [
                {"type": "construct", "entity_ids": [1, 2], "template": "structures/athen/house", "position": [24, 18]},
                {"type": "train", "entity_ids": [10], "template": "units/athen/support_female_citizen", "count": 1},
                {"type": "wait"},
            ], 1,
        ),
        _case(
            "probe-econ-003", "economy",
            "Avoid an imminent population block by constructing a house at the safe marked position.",
            {"resources": {"food": 260, "wood": 180}, "population": {"used": 19, "limit": 20}, "workers": [1, 2], "safe_build_position": [24, 18]},
            [
                {"type": "train", "entity_ids": [10], "template": "units/athen/support_female_citizen", "count": 1},
                {"type": "construct", "entity_ids": [1, 2], "template": "structures/athen/house", "position": [24, 18]},
                {"type": "gather", "entity_ids": [1, 2], "target_entity_id": 102},
            ], 1,
        ),
        _case(
            "probe-tech-001", "building-tech",
            "Advance to Town Phase now that the resource and building requirements are satisfied.",
            {"phase": "village", "resources": {"food": 900, "wood": 850}, "requirements_met": True, "civic_centre": [10]},
            [
                {"type": "research", "entity_ids": [10], "template": "phase_town"},
                {"type": "construct", "entity_ids": [1, 2], "template": "structures/athen/storehouse", "position": [31, 22]},
                {"type": "wait"},
            ], 0,
        ),
        _case(
            "probe-tech-002", "building-tech",
            "Unlock infantry production by constructing the missing barracks at the safe marked position.",
            {"phase": "village", "resources": {"wood": 420}, "structures": {"house": 2, "barracks": 0}, "workers": [3, 4], "safe_build_position": [40, 26]},
            [
                {"type": "construct", "entity_ids": [3, 4], "template": "structures/athen/barracks", "position": [40, 26]},
                {"type": "construct", "entity_ids": [3, 4], "template": "structures/athen/house", "position": [40, 26]},
                {"type": "research", "entity_ids": [10], "template": "phase_town"},
            ], 0,
        ),
        _case(
            "probe-tech-003", "building-tech",
            "Train two spearmen from the barracks to meet the infantry requirement.",
            {"resources": {"food": 340, "wood": 260, "metal": 40}, "population": {"used": 22, "limit": 30}, "barracks": [20], "infantry_required": 2},
            [
                {"type": "train", "entity_ids": [20], "template": "units/athen/infantry_spearman_b", "count": 2},
                {"type": "train", "entity_ids": [20], "template": "units/athen/cavalry_javelinist_b", "count": 2},
                {"type": "wait"},
            ], 0,
        ),
        _case(
            "probe-scout-001", "scouting",
            "Reveal the unexplored northern resource cluster with the available scout.",
            {"scouts": [30], "unexplored_targets": {"north": [72, 88], "south": [18, 9]}, "priority_hint": "north resource cluster"},
            [
                {"type": "scout", "entity_ids": [30], "position": [18, 9]},
                {"type": "scout", "entity_ids": [30], "position": [72, 88]},
                {"type": "wait"},
            ], 1,
        ),
        _case(
            "probe-scout-002", "scouting",
            "Preserve the wounded scout by retreating to the civic centre rally point.",
            {"scouts": {"30": {"health_fraction": 0.18, "under_attack": True}}, "enemy_units_nearby": 4, "safe_rally_point": [12, 14]},
            [
                {"type": "scout", "entity_ids": [30], "position": [80, 80]},
                {"type": "attack", "entity_ids": [30], "target_entity_id": 91},
                {"type": "retreat", "entity_ids": [30], "position": [12, 14]},
            ], 2,
        ),
        _case(
            "probe-scout-003", "scouting",
            "Investigate the reported enemy expansion before committing the army.",
            {"scouts": [31], "event_log": [{"type": "enemy_foundation_seen", "position": [91, 44], "confidence": 0.72}], "army": [40, 41, 42]},
            [
                {"type": "attack", "entity_ids": [40, 41, 42], "target_entity_id": 99},
                {"type": "scout", "entity_ids": [31], "position": [91, 44]},
                {"type": "scout", "entity_ids": [31], "position": [10, 90]},
            ], 1,
        ),
        _case(
            "probe-combat-001", "combat",
            "Defend the civic centre from the incoming raid using the nearby infantry.",
            {"infantry": [40, 41, 42, 43], "civic_centre": 10, "events": {"raid_target": 10}, "enemy_raiders": 3},
            [
                {"type": "defend", "entity_ids": [40, 41, 42, 43], "target_entity_id": 10},
                {"type": "scout", "entity_ids": [40], "position": [90, 90]},
                {"type": "attack", "entity_ids": [40, 41, 42, 43], "target_entity_id": 80},
            ], 0,
        ),
        _case(
            "probe-combat-002", "combat",
            "Destroy the exposed enemy siege engine before it reaches firing range.",
            {"army": [40, 41, 42, 43, 44], "enemy": {"siege_entity": 95, "escort_count": 1}, "own_army_advantage": True},
            [
                {"type": "defend", "entity_ids": [40, 41, 42, 43, 44], "target_entity_id": 10},
                {"type": "attack", "entity_ids": [40, 41, 42, 43, 44], "target_entity_id": 95},
                {"type": "retreat", "entity_ids": [40, 41, 42, 43, 44], "position": [12, 14]},
            ], 1,
        ),
        _case(
            "probe-combat-003", "combat",
            "Retreat the outnumbered army to the fortified rally point to avoid losing it.",
            {"army": [40, 41, 42], "own_strength": 3, "enemy_strength": 11, "fortified_rally_point": [22, 20]},
            [
                {"type": "attack", "entity_ids": [40, 41, 42], "target_entity_id": 90},
                {"type": "retreat", "entity_ids": [40, 41, 42], "position": [22, 20]},
                {"type": "wait"},
            ], 1,
        ),
        _case(
            "probe-adapt-001", "adaptive",
            "The assigned wood source is depleted. Reassign the workers to the replacement wood source.",
            {"workers": [1, 2, 3], "resource_nodes": {"102": {"type": "wood", "remaining": 0}, "108": {"type": "wood", "remaining": 1800}}, "events": {"resource_depleted": 102}},
            [
                {"type": "gather", "entity_ids": [1, 2, 3], "target_entity_id": 102},
                {"type": "gather", "entity_ids": [1, 2, 3], "target_entity_id": 108},
                {"type": "wait"},
            ], 1,
        ),
        _case(
            "probe-adapt-002", "adaptive",
            "The only barracks was destroyed. Rebuild it at the fallback construction position.",
            {"workers": [3, 4], "resources": {"wood": 510}, "structures": {"barracks": 0}, "events": {"structure_destroyed": {"template": "structures/athen/barracks"}}, "fallback_position": [37, 29]},
            [
                {"type": "construct", "entity_ids": [3, 4], "template": "structures/athen/barracks", "position": [37, 29]},
                {"type": "construct", "entity_ids": [3, 4], "template": "structures/athen/house", "position": [37, 29]},
                {"type": "wait"},
            ], 0,
        ),
        _case(
            "probe-adapt-003", "adaptive",
            "Enemy cavalry replaced its infantry force. Train three spearmen as the available counter unit.",
            {"barracks": [20], "resources": {"food": 500, "wood": 360}, "population": {"used": 31, "limit": 40}, "enemy_composition": {"cavalry": 8, "infantry": 1}, "events": {"composition_changed": True}},
            [
                {"type": "train", "entity_ids": [20], "template": "units/athen/infantry_spearman_b", "count": 3},
                {"type": "train", "entity_ids": [20], "template": "units/athen/infantry_archer_b", "count": 3},
                {"type": "attack", "entity_ids": [40, 41], "target_entity_id": 90},
            ], 0,
        ),
    )


class ProbeEnvironment:
    """One-decision environment with a hidden expected macro action."""

    def __init__(self, case: ProbeCase):
        self.case = case
        self.step_number = 0
        self.success = False
        self.last_action: Mapping[str, Any] = {"type": "wait"}

    def _observation(self, evaluator: bool = False) -> Dict[str, Any]:
        value = {
            "schema_version": "0.1",
            "episode_id": self.case.id,
            "step": self.step_number,
            "game_time_ms": self.step_number * 200,
            "track": "symbolic",
            "objective": self.case.objective,
            "player": copy.deepcopy(self.case.state),
            "opponent": {},
            "map": {},
            "entities": [],
            "events": {"probe_success": self.success},
            "event_log": [],
            "action_candidates": copy.deepcopy(list(self.case.candidates)),
        }
        if evaluator:
            value["expected_action"] = copy.deepcopy(self.case.expected_action)
        return value

    def reset(self, task: TaskSpec, seed: int) -> Mapping[str, Any]:
        self.step_number = 0
        self.success = False
        self.last_action = {"type": "wait"}
        return self._observation()

    def legal_actions(self) -> Sequence[str]:
        return tuple(dict.fromkeys(str(action["type"]) for action in self.case.candidates))

    def validate_action(self, action: Mapping[str, Any]) -> bool:
        return any(dict(action) == dict(candidate) for candidate in self.case.candidates)

    def step(self, action: Mapping[str, Any]) -> Transition:
        self.step_number += 1
        self.last_action = dict(action)
        self.success = dict(action) == dict(self.case.expected_action)
        category_reward = 1.0 if self.success else 0.0
        info = {"probe_success": self.success, "survival_reward": category_reward}
        if self.case.category == "economy":
            info["economy_reward"] = category_reward
        if self.case.category in {"combat", "adaptive"}:
            info["military_reward"] = category_reward
        if self.case.category == "adaptive":
            info["recovery_success"] = self.success
        return Transition(self._observation(), reward=category_reward, terminated=True, info=info)

    def evaluation_observation(self) -> Mapping[str, Any]:
        return self._observation(evaluator=True)

    def apply_perturbation(self, perturbation: Mapping[str, Any]) -> None:
        return None

    def export_replay(self, destination: Path):
        return None

    def close(self) -> None:
        return None


def summarize_probe_results(summaries: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    rows = list(summaries)
    by_category: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        category = str(row["category"])
        entry = by_category.setdefault(category, {"correct": 0, "total": 0})
        entry["total"] += 1
        entry["correct"] += int(bool(row["success"]))
    for entry in by_category.values():
        entry["accuracy"] = round(entry["correct"] / max(1, entry["total"]), 4)
    total = len(rows)
    correct = sum(int(bool(row["success"])) for row in rows)
    actions = sum(int(row["metrics"]["action_count"]) for row in rows)
    illegal = sum(int(row["metrics"]["illegal_action_count"]) for row in rows)
    return {
        "suite": "0AD-Bench symbolic model compatibility v0.1",
        "probe_count": total,
        "correct": correct,
        "accuracy": round(correct / max(1, total), 4),
        "legal_action_rate": round(1.0 - illegal / max(1, actions), 4),
        "mean_latency_ms": round(sum(float(row["metrics"]["mean_latency_ms"]) for row in rows) / max(1, total), 2),
        "total_tokens": sum(int(row["metrics"]["total_tokens"]) for row in rows),
        "categories": by_category,
    }

