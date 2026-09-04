"""Typed benchmark records shared by runners, agents, and adapters."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Sequence


@dataclass(frozen=True)
class TaskSpec:
    id: str
    version: str
    split: str
    category: str
    title: str
    description: str
    tracks: Sequence[str]
    scenario: Mapping[str, Any]
    budget: Mapping[str, Any]
    goal: Mapping[str, Any]
    perturbations: Sequence[Mapping[str, Any]] = field(default_factory=tuple)
    tags: Sequence[str] = field(default_factory=tuple)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TaskSpec":
        return cls(
            id=str(value["id"]),
            version=str(value["version"]),
            split=str(value.get("split", "development")),
            category=str(value["category"]),
            title=str(value["title"]),
            description=str(value["description"]),
            tracks=tuple(value["tracks"]),
            scenario=dict(value["scenario"]),
            budget=dict(value["budget"]),
            goal=dict(value["goal"]),
            perturbations=tuple(value.get("perturbations", [])),
            tags=tuple(value.get("tags", [])),
        )

    def to_dict(self) -> Dict[str, Any]:
        return json.loads(json.dumps(asdict(self)))


@dataclass(frozen=True)
class AgentDecision:
    action: Mapping[str, Any]
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    rationale: str = ""

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass(frozen=True)
class Transition:
    observation: Mapping[str, Any]
    reward: float = 0.0
    terminated: bool = False
    truncated: bool = False
    info: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class EpisodeMetrics:
    task_reward: float = 0.0
    economy_reward: float = 0.0
    military_reward: float = 0.0
    survival_reward: float = 0.0
    action_count: int = 0
    illegal_action_count: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms_total: float = 0.0
    recovery_success: bool = False
    completed_steps: int = 0
    wall_clock_seconds: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def mean_latency_ms(self) -> float:
        return self.latency_ms_total / max(1, self.action_count)

    @property
    def illegal_action_rate(self) -> float:
        return self.illegal_action_count / max(1, self.action_count)

    def to_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        value.update(
            total_tokens=self.total_tokens,
            mean_latency_ms=self.mean_latency_ms,
            illegal_action_rate=self.illegal_action_rate,
        )
        return value
