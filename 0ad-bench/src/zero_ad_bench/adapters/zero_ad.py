"""Boundary between the official zero_ad RL client and 0AD-Bench."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

from ..models import TaskSpec, Transition


def _entity_id(unit: Any) -> int:
    if hasattr(unit, "id"):
        candidate = unit.id
        return int(candidate() if callable(candidate) else candidate)
    return int(unit.data.get("id", unit.data.get("entity", -1)))


def _position(unit: Any) -> Optional[list]:
    if "position" not in unit.data:
        return None
    value = unit.position()
    return [round(float(value[0]), 3), round(float(value[1]), 3)]


def _player(state: Any, player_id: int) -> dict:
    source = state.data["players"][player_id]
    units = list(state.units(owner=player_id))
    structures = [unit for unit in units if unit.type().startswith("structures/")]
    mobile = [unit for unit in units if unit.type().startswith("units/")]
    counts = {}
    for structure in structures:
        name = structure.type().split("/")[-1]
        counts[name] = counts.get(name, 0) + 1
    unit_counts = {}
    for unit in mobile:
        name = unit.type().split("/")[-1]
        unit_counts[name] = unit_counts.get(name, 0) + 1
    statistics = dict(source.get("statistics", {}))
    statistics.setdefault("worker_count", sum(1 for unit in mobile if "support_civilian" in unit.type() or "infantry" in unit.type()))
    visible_structures = {}
    for structure in structures:
        name = structure.type().split("/")[-1]
        visible_structures[name] = visible_structures.get(name, 0) + 1
    military_templates = {unit.type() for unit in mobile if "support_civilian" not in unit.type()}
    return {
        "id": player_id,
        "state": source.get("state", "unknown"),
        "phase": source.get("phase", "unknown"),
        "resources": source.get("resourceCounts", {}),
        "resource_gatherers": source.get("resourceGatherers", {}),
        "population": {"used": source.get("popCount", 0), "limit": source.get("popLimit", 0)},
        "units": unit_counts,
        "unit_count": len(mobile),
        "structures": counts,
        "structure_count": len(structures),
        "statistics": statistics,
        "visible_structure_count": len(structures),
        "visible_structures": visible_structures,
        "visible_military_templates": len(military_templates),
    }


def normalize_state(
    state: Any,
    episode_id: str,
    step: int,
    player_id: int = 1,
    opponent_id: int = 2,
    track: str = "symbolic",
    frame_path: Optional[str] = None,
) -> dict:
    entities = []
    for owner in (0, player_id, opponent_id):
        for unit in state.units(owner=owner):
            entities.append(
                {
                    "id": _entity_id(unit),
                    "owner": owner,
                    "template": unit.type(),
                    "position": _position(unit),
                    "visible": True,
                }
            )
    observation = {
        "schema_version": "0.1",
        "episode_id": episode_id,
        "step": step,
        "game_time_ms": int(state.data.get("timeElapsed", 0)),
        "track": track,
        "player": _player(state, player_id),
        "opponent": _player(state, opponent_id),
        "map": state.data.get("benchmarkMap", state.data.get("map", {})),
        "entities": entities,
        "events": state.data.get("benchmarkEvents", {}),
        "event_log": state.data.get("events", []),
    }
    if frame_path is not None:
        observation["frame_path"] = frame_path
    return observation


class ZeroADEnvironment:
    """Reference environment around an already running official RL endpoint.

    The reset payload factory supplies a Release-matched scenario dictionary.
    Perturbations are delegated to an optional deterministic engine hook.
    """

    def __init__(
        self,
        game: Any,
        actions_module: Any,
        reset_payload_factory: Callable[[TaskSpec, int], Mapping[str, Any]],
        player_id: int = 1,
        opponent_id: int = 2,
        track: str = "symbolic",
        perturbation_hook: Optional[Callable[[Mapping[str, Any], Any], None]] = None,
        replay_exporter: Optional[Callable[[Path], Optional[Path]]] = None,
        close_hook: Optional[Callable[[], None]] = None,
        frame_provider: Optional[Callable[[int], str]] = None,
    ):
        self.game = game
        self.actions = actions_module
        self.reset_payload_factory = reset_payload_factory
        self.player_id = player_id
        self.opponent_id = opponent_id
        self.track = track
        self.perturbation_hook = perturbation_hook
        self.replay_exporter = replay_exporter
        self.close_hook = close_hook
        self.frame_provider = frame_provider
        self.state = None
        self.step_number = 0
        self.episode_id = "uninitialized"

    def reset(self, task: TaskSpec, seed: int) -> Mapping[str, Any]:
        self.episode_id = f"{task.id}-{seed}"
        self.step_number = 0
        payload = dict(self.reset_payload_factory(task, seed))
        self.state = self.game.reset(json.dumps(payload), save_replay=True, player_id=self.player_id)
        return self._observation()

    def _observation(self) -> Mapping[str, Any]:
        frame = self.frame_provider(self.step_number) if self.frame_provider else None
        full = normalize_state(
            self.state,
            self.episode_id,
            self.step_number,
            self.player_id,
            self.opponent_id,
            self.track,
            frame,
        )
        if self.track == "symbolic":
            full.pop("frame_path", None)
            return full
        if self.track == "hybrid":
            if frame is None:
                raise RuntimeError("Hybrid track requires a frame provider")
            return full
        if self.track == "vision":
            if frame is None:
                raise RuntimeError("Vision track requires a frame provider")
            return {
                "schema_version": "0.1",
                "episode_id": self.episode_id,
                "step": self.step_number,
                "game_time_ms": full["game_time_ms"],
                "track": "vision",
                "frame_path": frame,
                "events": {},
            }
        raise ValueError(f"Unknown track: {self.track}")

    def evaluation_observation(self) -> Mapping[str, Any]:
        return normalize_state(
            self.state,
            self.episode_id,
            self.step_number,
            self.player_id,
            self.opponent_id,
            self.track,
        )

    def legal_actions(self) -> Sequence[str]:
        return (
            "wait", "gather", "train", "construct", "research",
            "walk", "scout", "attack", "defend", "retreat",
        )

    def _units(self, entity_ids: Sequence[int]) -> list:
        wanted = set(int(value) for value in entity_ids)
        return [unit for unit in self.state.units(owner=self.player_id) if _entity_id(unit) in wanted]

    def validate_action(self, action: Mapping[str, Any]) -> bool:
        action_type = action.get("type", "wait")
        if action_type == "wait":
            return True
        entity_ids = action.get("entity_ids", [])
        if not entity_ids or len(self._units(entity_ids)) != len(set(entity_ids)):
            return False
        if action_type in {"gather", "attack", "defend"}:
            return self._target(action.get("target_entity_id", -1)) is not None
        if action_type in {"walk", "scout", "retreat", "construct"}:
            position = action.get("position", [])
            if len(position) != 2 or not all(isinstance(value, (int, float)) for value in position):
                return False
        if action_type in {"train", "construct", "research"} and not action.get("template"):
            return False
        return True

    def _target(self, entity_id: int) -> Any:
        for owner in (0, self.player_id, self.opponent_id):
            for unit in self.state.units(owner=owner):
                if _entity_id(unit) == int(entity_id):
                    return unit
        return None

    def _translate(self, action: Mapping[str, Any]) -> list:
        action_type = action.get("type", "wait")
        units = self._units(action.get("entity_ids", []))
        if action_type == "wait":
            return []
        if action_type == "gather":
            target = self._target(action["target_entity_id"])
            return [self.actions.gather(units, target)] if units and target else []
        if action_type == "train":
            return [self.actions.train(units, action["template"], int(action.get("count", 1)))] if units else []
        if action_type == "construct":
            x, z = action["position"]
            return [self.actions.construct(units, action["template"], float(x), float(z))] if units else []
        if action_type in {"walk", "scout", "retreat"}:
            x, z = action["position"]
            return [self.actions.walk(units, float(x), float(z))] if units else []
        if action_type in {"attack", "defend"}:
            target = self._target(action["target_entity_id"])
            return [self.actions.attack(units, target)] if units and target else []
        if action_type == "research" and hasattr(self.actions, "research"):
            return [self.actions.research(units, action["template"])] if units else []
        return []

    def step(self, action: Mapping[str, Any]) -> Transition:
        self.state = self.game.step(self._translate(action))
        self.step_number += 1
        observation = self._observation()
        player_state = observation["player"]["state"]
        opponent_state = observation["opponent"]["state"]
        terminated = player_state != "active" or opponent_state != "active"
        return Transition(observation, terminated=terminated, info={"survival_reward": 1.0 if player_state == "active" else 0.0})

    def apply_perturbation(self, perturbation: Mapping[str, Any]) -> None:
        if self.perturbation_hook is None:
            raise RuntimeError("This task requires a deterministic perturbation hook")
        self.perturbation_hook(perturbation, self.state)

    def export_replay(self, destination: Path) -> Optional[Path]:
        return self.replay_exporter(destination) if self.replay_exporter else None

    def close(self) -> None:
        if self.close_hook:
            self.close_hook()
