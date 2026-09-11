"""0AD-Bench v0.1 public API."""

from .leaderboard import aggregate_leaderboard, leaderboard_markdown
from .models import AgentDecision, EpisodeMetrics, TaskSpec, Transition
from .probes import ProbeCase, ProbeEnvironment, default_probe_cases, summarize_probe_results
from .registry import TaskRegistry
from .runner import BenchmarkRunner
from .scoring import ScoreBreakdown, score_episode
from .trajectory import EpisodeRecorder

__version__ = "0.1.0"

__all__ = [
    "AgentDecision",
    "BenchmarkRunner",
    "EpisodeMetrics",
    "EpisodeRecorder",
    "ProbeCase",
    "ProbeEnvironment",
    "ScoreBreakdown",
    "TaskRegistry",
    "TaskSpec",
    "Transition",
    "aggregate_leaderboard",
    "default_probe_cases",
    "leaderboard_markdown",
    "score_episode",
    "summarize_probe_results",
]
