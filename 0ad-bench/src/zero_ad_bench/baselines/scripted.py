"""Small deterministic baseline for runner checks and lower-bound results."""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

from ..models import AgentDecision


class ScriptedBaseline:
    def __init__(
        self,
        policy: Callable[[Mapping[str, Any], Sequence[str], Mapping[str, Any]], Mapping[str, Any]],
        name: str = "scripted-baseline",
    ):
        self.policy = policy
        self.name = name

    def act(self, observation, legal_actions, context) -> AgentDecision:
        action = self.policy(observation, legal_actions, context)
        return AgentDecision(action=action, rationale="deterministic scripted baseline")
