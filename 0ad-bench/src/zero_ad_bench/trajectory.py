"""Benchmark trajectory writer with stable, inspectable artifacts."""

from __future__ import annotations

import json
import pathlib
import shutil
from typing import Any, Mapping, Optional


class EpisodeRecorder:
    def __init__(self, root: pathlib.Path, episode_id: str):
        self.directory = pathlib.Path(root) / episode_id
        self.observations = self.directory / "observations"
        self.evaluator_observations = self.directory / "evaluator_observations"
        self.replay = self.directory / "replay"
        self.observations.mkdir(parents=True, exist_ok=False)
        self.evaluator_observations.mkdir(parents=True, exist_ok=True)
        self.replay.mkdir(parents=True, exist_ok=True)
        self._actions = (self.directory / "actions.jsonl").open("w", encoding="utf-8")
        self._reasoning = (self.directory / "reasoning.jsonl").open("w", encoding="utf-8")
        self._rewards = (self.directory / "rewards.jsonl").open("w", encoding="utf-8")

    @staticmethod
    def _write_json(path: pathlib.Path, value: Mapping[str, Any]) -> None:
        path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    @staticmethod
    def _append(stream, value: Mapping[str, Any]) -> None:
        stream.write(json.dumps(value, sort_keys=True) + "\n")
        stream.flush()

    def write_metadata(self, metadata: Mapping[str, Any]) -> None:
        self._write_json(self.directory / "metadata.json", metadata)

    def write_observation(self, step: int, observation: Mapping[str, Any]) -> None:
        self._write_json(self.observations / f"{step:06d}.json", observation)

    def write_evaluator_observation(self, step: int, observation: Mapping[str, Any]) -> None:
        self._write_json(self.evaluator_observations / f"{step:06d}.json", observation)

    def write_action(self, step: int, action: Mapping[str, Any], legal: bool) -> None:
        self._append(self._actions, {"step": step, "legal": legal, "action": action})

    def write_reasoning(self, step: int, rationale: str) -> None:
        self._append(self._reasoning, {"step": step, "rationale": rationale})

    def write_reward(self, step: int, reward: float, info: Mapping[str, Any]) -> None:
        self._append(self._rewards, {"step": step, "reward": reward, "info": dict(info)})

    def write_summary(self, summary: Mapping[str, Any]) -> None:
        self._write_json(self.directory / "summary.json", summary)

    def add_replay(self, source: pathlib.Path) -> Optional[pathlib.Path]:
        source = pathlib.Path(source)
        if not source.exists():
            return None
        destination = self.replay / source.name
        if source.is_dir():
            shutil.copytree(source, destination, dirs_exist_ok=True)
        else:
            shutil.copy2(source, destination)
        return destination

    def close(self) -> None:
        for stream in (self._actions, self._reasoning, self._rewards):
            if not stream.closed:
                stream.close()

    def __enter__(self) -> "EpisodeRecorder":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
