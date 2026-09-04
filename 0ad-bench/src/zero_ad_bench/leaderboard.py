"""Deterministic aggregation for benchmark episode summaries."""

from __future__ import annotations

import statistics
from collections import defaultdict
from typing import Any, Iterable, List, Mapping


def aggregate_leaderboard(summaries: Iterable[Mapping[str, Any]]) -> List[dict]:
    grouped = defaultdict(list)
    for summary in summaries:
        grouped[(str(summary["agent"]), str(summary.get("track", "symbolic")))].append(summary)
    rows = []
    for (agent, track), episodes in grouped.items():
        scores = [float(item["score"]["total"]) for item in episodes]
        category_values = defaultdict(list)
        for item in episodes:
            category_values[str(item.get("category", "unknown"))].append(float(item["score"]["total"]))
        category_scores = {key: round(statistics.fmean(values), 4) for key, values in sorted(category_values.items())}
        actions = [int(item["metrics"]["action_count"]) for item in episodes]
        tokens = [int(item["metrics"]["total_tokens"]) for item in episodes]
        latency = [float(item["metrics"]["mean_latency_ms"]) for item in episodes]
        illegal = [float(item["metrics"]["illegal_action_rate"]) for item in episodes]
        rows.append(
            {
                "agent": agent,
                "track": track,
                "episodes": len(episodes),
                "mean_score": round(statistics.fmean(category_scores.values()), 4),
                "median_score": round(statistics.median(scores), 4),
                "success_rate": round(sum(bool(item["success"]) for item in episodes) / len(episodes), 4),
                "mean_actions": round(statistics.fmean(actions), 2),
                "mean_tokens": round(statistics.fmean(tokens), 2),
                "mean_latency_ms": round(statistics.fmean(latency), 2),
                "illegal_action_rate": round(statistics.fmean(illegal), 4),
                "category_scores": category_scores,
            }
        )
    return sorted(rows, key=lambda row: (row["track"], -row["mean_score"], row["agent"]))


def leaderboard_markdown(rows: Iterable[Mapping[str, Any]]) -> str:
    lines = [
        "| Rank | Track | Agent | Episodes | Mean score | Success rate | Mean actions | Mean tokens | Mean latency ms | Illegal rate |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(rows, start=1):
        lines.append(
            f"| {rank} | {row['track']} | {row['agent']} | {row['episodes']} | {row['mean_score']:.2f} | "
            f"{row['success_rate']:.1%} | {row['mean_actions']:.2f} | {row['mean_tokens']:.2f} | "
            f"{row['mean_latency_ms']:.2f} | {row['illegal_action_rate']:.1%} |"
        )
    return "\n".join(lines)
