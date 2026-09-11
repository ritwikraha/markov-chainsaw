"""Reference 0AD-Bench v0.1 episode scoring."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping

from .models import EpisodeMetrics


@dataclass(frozen=True)
class ScoreBreakdown:
    total: float
    task: float
    economy: float
    military: float
    survival: float
    action_penalty: float
    token_penalty: float
    latency_penalty: float
    illegal_action_penalty: float

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


def _unit(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def score_episode(
    metrics: EpisodeMetrics,
    budget: Mapping[str, Any],
) -> ScoreBreakdown:
    """Score an episode on a 0 to 100 scale.

    Positive components sum to 100 points. Efficiency penalties can remove up
    to 20 points. Each cost is normalized by a task-specific budget.
    """
    max_actions = max(1, int(budget.get("max_agent_actions", budget.get("max_steps", 1))))
    max_tokens = max(1, int(budget.get("max_tokens", 1)))
    max_latency_ms = max(1.0, float(budget.get("max_mean_latency_ms", 30_000)))

    task = 50.0 * _unit(metrics.task_reward)
    economy = 20.0 * _unit(metrics.economy_reward)
    military = 15.0 * _unit(metrics.military_reward)
    survival = 15.0 * _unit(metrics.survival_reward)
    action_penalty = 5.0 * _unit(metrics.action_count / max_actions)
    token_penalty = 5.0 * _unit(metrics.total_tokens / max_tokens)
    latency_penalty = 5.0 * _unit(metrics.mean_latency_ms / max_latency_ms)
    illegal_action_penalty = 5.0 * _unit(metrics.illegal_action_rate)
    total = max(
        0.0,
        task + economy + military + survival
        - action_penalty - token_penalty - latency_penalty - illegal_action_penalty,
    )
    return ScoreBreakdown(
        total=round(total, 4),
        task=round(task, 4),
        economy=round(economy, 4),
        military=round(military, 4),
        survival=round(survival, 4),
        action_penalty=round(action_penalty, 4),
        token_penalty=round(token_penalty, 4),
        latency_penalty=round(latency_penalty, 4),
        illegal_action_penalty=round(illegal_action_penalty, 4),
    )
