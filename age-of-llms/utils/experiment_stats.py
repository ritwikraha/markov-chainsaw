"""Comparable player statistics and deterministic turn-limit scoring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


METRIC_LABELS = {
    "state": "Game state",
    "phase": "Phase",
    "population": "Population",
    "population_limit": "Population limit",
    "workers": "Economic workers",
    "military": "Military units",
    "structures": "Structures",
    "food": "Food",
    "wood": "Wood",
    "stone": "Stone",
    "metal": "Metal",
    "resources": "Total banked resources",
    "score": "Turn-limit score",
}


@dataclass(frozen=True)
class ExperimentOutcome:
    """Final comparison for two 0 A.D. players."""

    winner: str
    result_type: str
    reason: str
    gemma: dict[str, Any]
    opponent: dict[str, Any]

    def rows(self) -> list[dict[str, Any]]:
        return [
            {
                "Metric": label,
                "Gemma": self.gemma[key],
                "Petra": self.opponent[key],
            }
            for key, label in METRIC_LABELS.items()
        ]

    def markdown(self) -> str:
        lines = ["| Metric | Gemma | Petra |", "|---|---:|---:|"]
        for row in self.rows():
            lines.append(f"| {row['Metric']} | {row['Gemma']} | {row['Petra']} |")
        return "\n".join(lines)


def _player_snapshot(state: Any, player_id: int) -> dict[str, Any]:
    player = state.data["players"][player_id]
    units = list(state.units(owner=player_id))
    mobile = [unit for unit in units if unit.type().startswith("units/")]
    structures = [unit for unit in units if unit.type().startswith("structures/")]
    gatherers = player.get("resourceGatherers", {}) or {}
    workers = sum(int(value or 0) for value in gatherers.values())
    civilians = [
        unit
        for unit in mobile
        if "support_civilian" in unit.type() or "infantry" in unit.type()
    ]
    workers = max(workers, len(civilians))
    military = [unit for unit in mobile if "support_civilian" not in unit.type()]
    resources = player.get("resourceCounts", {}) or {}
    total_resources = sum(int(resources.get(name, 0) or 0) for name in ("food", "wood", "stone", "metal"))
    snapshot = {
        "state": player.get("state", "unknown"),
        "phase": player.get("phase", "unknown"),
        "population": int(player.get("popCount", len(mobile)) or 0),
        "population_limit": int(player.get("popLimit", 0) or 0),
        "workers": workers,
        "military": len(military),
        "structures": len(structures),
        "food": int(resources.get("food", 0) or 0),
        "wood": int(resources.get("wood", 0) or 0),
        "stone": int(resources.get("stone", 0) or 0),
        "metal": int(resources.get("metal", 0) or 0),
        "resources": total_resources,
    }
    snapshot["score"] = round(
        snapshot["population"]
        + 3 * snapshot["workers"]
        + 6 * snapshot["military"]
        + 8 * snapshot["structures"]
        + snapshot["resources"] / 100,
        1,
    )
    return snapshot


def _advantage_text(winner: dict[str, Any], loser: dict[str, Any]) -> str:
    comparisons = []
    for key, label in (
        ("military", "military units"),
        ("workers", "economic workers"),
        ("structures", "structures"),
        ("population", "population"),
        ("resources", "banked resources"),
    ):
        difference = winner[key] - loser[key]
        if difference > 0:
            comparisons.append((difference / max(1, loser[key]), f"{difference:g} more {label}"))
    comparisons.sort(reverse=True)
    return ", ".join(text for _, text in comparisons[:3]) or "the higher aggregate turn-limit score"


def compare_players(
    state: Any,
    gemma_id: int = 1,
    opponent_id: int = 2,
    gemma_name: str = "Gemma",
    opponent_name: str = "Petra",
) -> ExperimentOutcome:
    """Choose an official winner or a deterministic leader at the turn limit."""
    gemma = _player_snapshot(state, gemma_id)
    opponent = _player_snapshot(state, opponent_id)
    if gemma["state"] == "won" or opponent["state"] in {"defeated", "lost"}:
        winner, result_type = gemma_name, "conquest"
        reason = f"{gemma_name} won because the engine marked the opponent defeated."
    elif opponent["state"] == "won" or gemma["state"] in {"defeated", "lost"}:
        winner, result_type = opponent_name, "conquest"
        reason = f"{opponent_name} won because the engine marked {gemma_name} defeated."
    elif gemma["score"] > opponent["score"]:
        winner, result_type = gemma_name, "turn-limit leader"
        reason = f"{gemma_name} led at the turn limit with {_advantage_text(gemma, opponent)}."
    elif opponent["score"] > gemma["score"]:
        winner, result_type = opponent_name, "turn-limit leader"
        reason = f"{opponent_name} led at the turn limit with {_advantage_text(opponent, gemma)}."
    else:
        winner, result_type = "Draw", "turn-limit draw"
        reason = "Both sides had the same deterministic turn-limit score."
    return ExperimentOutcome(winner, result_type, reason, gemma, opponent)
