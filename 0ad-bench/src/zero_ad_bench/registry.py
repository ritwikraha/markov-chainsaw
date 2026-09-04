"""Versioned benchmark task discovery and validation."""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path
from typing import Iterable, List, Optional

from .models import TaskSpec


VALID_CATEGORIES = {"economy", "building-tech", "scouting", "combat", "adaptive"}
VALID_TRACKS = {"symbolic", "vision", "hybrid"}


def _merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


class TaskRegistry:
    def __init__(self, tasks: Iterable[TaskSpec], benchmark_version: str = "0.1"):
        self.benchmark_version = benchmark_version
        task_list = list(tasks)
        self._tasks = {task.id: task for task in task_list}
        if len(self._tasks) != len(task_list):
            raise ValueError("Task registry contains duplicate IDs")
        self.validate()

    @classmethod
    def default(cls) -> "TaskRegistry":
        resource = resources.files("zero_ad_bench").joinpath("data/tasks_v0_1.json")
        with resource.open(encoding="utf-8") as stream:
            return cls.from_dict(json.load(stream))

    @classmethod
    def from_path(cls, path: Path) -> "TaskRegistry":
        with Path(path).open(encoding="utf-8") as stream:
            return cls.from_dict(json.load(stream))

    @classmethod
    def from_dict(cls, value: dict) -> "TaskRegistry":
        defaults = value.get("defaults", {})
        return cls(
            [TaskSpec.from_dict(_merge(defaults, task)) for task in value["tasks"]],
            benchmark_version=str(value.get("benchmark_version", "0.1")),
        )

    def validate(self) -> None:
        if len(self._tasks) == 0:
            raise ValueError("Task registry is empty")
        for task in self._tasks.values():
            if task.version != self.benchmark_version:
                raise ValueError(f"Task {task.id} version does not match the registry")
            if task.split not in {"development", "evaluation"}:
                raise ValueError(f"Invalid split for {task.id}: {task.split}")
            if task.category not in VALID_CATEGORIES:
                raise ValueError(f"Invalid category for {task.id}: {task.category}")
            unknown_tracks = set(task.tracks) - VALID_TRACKS
            if unknown_tracks:
                raise ValueError(f"Invalid tracks for {task.id}: {sorted(unknown_tracks)}")
            if int(task.budget.get("max_steps", 0)) <= 0:
                raise ValueError(f"Task {task.id} must define a positive max_steps")
            if not task.goal.get("predicates"):
                raise ValueError(f"Task {task.id} has no goal predicates")
            for perturbation in task.perturbations:
                if int(perturbation["at_step"]) > int(task.budget["max_steps"]):
                    raise ValueError(f"Task {task.id} has a perturbation beyond max_steps")

    def get(self, task_id: str) -> TaskSpec:
        try:
            return self._tasks[task_id]
        except KeyError as exc:
            raise KeyError(f"Unknown task: {task_id}") from exc

    def list(
        self,
        category: Optional[str] = None,
        track: Optional[str] = None,
        split: Optional[str] = None,
    ) -> List[TaskSpec]:
        tasks = self._tasks.values()
        if category:
            tasks = (task for task in tasks if task.category == category)
        if track:
            tasks = (task for task in tasks if track in task.tracks)
        if split:
            tasks = (task for task in tasks if task.split == split)
        return sorted(tasks, key=lambda task: task.id)

    def summary(self) -> dict:
        categories = {category: 0 for category in sorted(VALID_CATEGORIES)}
        tracks = {track: 0 for track in sorted(VALID_TRACKS)}
        splits = {"development": 0, "evaluation": 0}
        for task in self._tasks.values():
            categories[task.category] += 1
            splits[task.split] += 1
            for track in task.tracks:
                tracks[track] += 1
        return {
            "benchmark_version": self.benchmark_version,
            "task_count": len(self._tasks),
            "categories": categories,
            "tracks": tracks,
            "splits": splits,
        }
