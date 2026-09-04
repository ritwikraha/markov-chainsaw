"""Declarative task-goal evaluation over episode observations."""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence


def resolve_path(value: Mapping[str, Any], path: str) -> Any:
    current: Any = value
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def compare(actual: Any, operator: str, expected: Any) -> bool:
    if actual is None:
        return False
    if operator == "eq":
        return actual == expected
    if operator == "ne":
        return actual != expected
    if operator == "gte":
        return actual >= expected
    if operator == "lte":
        return actual <= expected
    if operator == "gt":
        return actual > expected
    if operator == "lt":
        return actual < expected
    if operator == "contains":
        return expected in actual
    raise ValueError(f"Unsupported predicate operator: {operator}")


def evaluate_predicate(
    predicate: Mapping[str, Any],
    observations: Sequence[Mapping[str, Any]],
) -> bool:
    scope = predicate.get("scope", "final")
    values = [resolve_path(observation, str(predicate["path"])) for observation in observations]
    if not values:
        return False
    if scope == "final":
        candidates = values[-1:]
    elif scope == "any":
        candidates = values
    elif scope == "max":
        candidates = [max(value for value in values if value is not None)] if any(value is not None for value in values) else []
    elif scope == "min":
        candidates = [min(value for value in values if value is not None)] if any(value is not None for value in values) else []
    else:
        raise ValueError(f"Unsupported predicate scope: {scope}")
    return any(compare(value, str(predicate["op"]), predicate["value"]) for value in candidates)


def task_reward(goal: Mapping[str, Any], observations: Sequence[Mapping[str, Any]]) -> float:
    predicates = list(goal.get("predicates", []))
    if not predicates:
        return 0.0
    results = [evaluate_predicate(predicate, observations) for predicate in predicates]
    aggregation = goal.get("aggregation", "all")
    if aggregation == "all":
        return 1.0 if all(results) else sum(results) / len(results)
    if aggregation == "any":
        return 1.0 if any(results) else 0.0
    if aggregation == "fraction":
        return sum(results) / len(results)
    raise ValueError(f"Unsupported goal aggregation: {aggregation}")
