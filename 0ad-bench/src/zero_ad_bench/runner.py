"""Reference closed-loop episode runner."""

from __future__ import annotations

import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Optional, Protocol, Sequence

from .models import AgentDecision, EpisodeMetrics, TaskSpec, Transition
from .predicates import task_reward
from .scoring import score_episode
from .trajectory import EpisodeRecorder


class BenchmarkEnvironment(Protocol):
    def reset(self, task: TaskSpec, seed: int) -> Mapping[str, Any]: ...
    def legal_actions(self) -> Sequence[str]: ...
    def validate_action(self, action: Mapping[str, Any]) -> bool: ...
    def step(self, action: Mapping[str, Any]) -> Transition: ...
    def apply_perturbation(self, perturbation: Mapping[str, Any]) -> None: ...
    def export_replay(self, destination: Path) -> Optional[Path]: ...
    def close(self) -> None: ...


class BenchmarkAgent(Protocol):
    name: str
    def act(
        self,
        observation: Mapping[str, Any],
        legal_actions: Sequence[str],
        context: Mapping[str, Any],
    ) -> AgentDecision: ...


class BenchmarkRunner:
    def __init__(self, output_root: Path):
        self.output_root = Path(output_root)

    @staticmethod
    def _is_legal(action: Mapping[str, Any], legal_actions: Sequence[str]) -> bool:
        if not isinstance(action, Mapping) or action.get("type") not in set(legal_actions):
            return False
        required = {
            "gather": {"entity_ids", "target_entity_id"},
            "train": {"entity_ids", "template", "count"},
            "construct": {"entity_ids", "template", "position"},
            "research": {"entity_ids", "template"},
            "walk": {"entity_ids", "position"},
            "scout": {"entity_ids", "position"},
            "attack": {"entity_ids", "target_entity_id"},
            "defend": {"entity_ids", "target_entity_id"},
            "retreat": {"entity_ids", "position"},
        }
        return required.get(str(action.get("type")), set()).issubset(action)

    def run(
        self,
        task: TaskSpec,
        agent: BenchmarkAgent,
        environment: BenchmarkEnvironment,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
    ) -> Mapping[str, Any]:
        seed = int(seed if seed is not None else task.scenario.get("seed", 0))
        episode_id = episode_id or f"{task.id}-{agent.name}-{seed}-{uuid.uuid4().hex[:8]}"
        metrics = EpisodeMetrics()
        observations = []
        started = time.perf_counter()
        max_steps = int(task.budget["max_steps"])
        decision_interval = max(1, int(task.budget.get("decision_interval", 1)))
        perturbations = {int(item["at_step"]): item for item in task.perturbations}
        final_perturbation_step = max(perturbations, default=0)

        with EpisodeRecorder(self.output_root, episode_id) as recorder:
            recorder.write_metadata(
                {
                    "schema_version": "0.1",
                    "benchmark_version": task.version,
                    "episode_id": episode_id,
                    "task": task.to_dict(),
                    "agent": agent.name,
                    "seed": seed,
                }
            )
            try:
                observation = environment.reset(task, seed)
                observations.append(observation)
                recorder.write_observation(0, observation)
                evaluator = getattr(environment, "evaluation_observation", lambda: observation)()
                evaluation_observations = [evaluator]
                recorder.write_evaluator_observation(0, evaluator)
                last_action: Mapping[str, Any] = {"type": "wait"}

                for step in range(1, max_steps + 1):
                    if step in perturbations:
                        environment.apply_perturbation(perturbations[step])
                    legal_actions = list(environment.legal_actions())
                    if step == 1 or (step - 1) % decision_interval == 0:
                        decision = agent.act(
                            observation,
                            legal_actions,
                            {"task_id": task.id, "step": step, "max_steps": max_steps},
                        )
                        metrics.action_count += 1
                        metrics.prompt_tokens += decision.prompt_tokens
                        metrics.completion_tokens += decision.completion_tokens
                        metrics.latency_ms_total += decision.latency_ms
                        legal = self._is_legal(decision.action, legal_actions)
                        validator = getattr(environment, "validate_action", None)
                        if legal and validator is not None:
                            legal = bool(validator(decision.action))
                        if not legal:
                            metrics.illegal_action_count += 1
                            last_action = {"type": "wait"}
                        else:
                            last_action = decision.action
                        recorder.write_action(step, decision.action, legal)
                        recorder.write_reasoning(step, decision.rationale)
                    else:
                        last_action = {"type": "wait"}

                    transition = environment.step(last_action)
                    observation = transition.observation
                    observations.append(observation)
                    evaluator = getattr(environment, "evaluation_observation", lambda: observation)()
                    evaluation_observations.append(evaluator)
                    metrics.completed_steps = step
                    recorder.write_observation(step, observation)
                    recorder.write_evaluator_observation(step, evaluator)
                    recorder.write_reward(step, transition.reward, transition.info)
                    metrics.economy_reward = max(metrics.economy_reward, float(transition.info.get("economy_reward", 0.0)))
                    metrics.military_reward = max(metrics.military_reward, float(transition.info.get("military_reward", 0.0)))
                    metrics.survival_reward = max(metrics.survival_reward, float(transition.info.get("survival_reward", 0.0)))
                    metrics.recovery_success = metrics.recovery_success or bool(transition.info.get("recovery_success", False))
                    if transition.terminated or transition.truncated:
                        break
                    if (
                        task.goal.get("terminate_on_success", True)
                        and step >= final_perturbation_step
                        and task_reward(task.goal, evaluation_observations) == 1.0
                    ):
                        break

                metrics.task_reward = task_reward(task.goal, evaluation_observations)
                metrics.wall_clock_seconds = time.perf_counter() - started
                breakdown = score_episode(metrics, task.budget)
                summary = {
                    "schema_version": "0.1",
                    "episode_id": episode_id,
                    "task_id": task.id,
                    "category": task.category,
                    "track": observation.get("track", "symbolic"),
                    "agent": agent.name,
                    "seed": seed,
                    "success": metrics.task_reward == 1.0,
                    "metrics": metrics.to_dict(),
                    "score": breakdown.to_dict(),
                    "final_observation": observation,
                    "final_evaluation_observation": evaluator,
                }
                replay = environment.export_replay(recorder.replay)
                if replay:
                    summary["replay"] = str(replay)
                recorder.write_summary(summary)
                return summary
            finally:
                environment.close()
