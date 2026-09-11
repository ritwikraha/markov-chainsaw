import json
from importlib import resources

import jsonschema

from zero_ad_bench import TaskRegistry, aggregate_leaderboard
from zero_ad_bench.smoke import EconomySmokeEnvironment


def schema(name):
    path = resources.files("zero_ad_bench").joinpath(f"schemas/{name}.schema.json")
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def validate(instance, value):
    validator = getattr(jsonschema, "Draft202012Validator", jsonschema.Draft7Validator)
    validator(value).validate(instance)


def test_task_action_and_observation_schemas():
    task = TaskRegistry.default().get("econ-001")
    validate(task.to_dict(), schema("task"))
    validate({"type": "gather", "entity_ids": [1], "target_entity_id": 2}, schema("action"))
    validate({"type": "pointer_click", "x": 400, "y": 300, "button": "left"}, schema("vision-action"))
    validate(EconomySmokeEnvironment()._observation(), schema("observation"))


def test_leaderboard_schema():
    summary = {
        "agent": "agent-a",
        "track": "symbolic",
        "category": "economy",
        "success": True,
        "score": {"total": 75},
        "metrics": {"action_count": 3, "total_tokens": 30, "mean_latency_ms": 10, "illegal_action_rate": 0},
    }
    row = aggregate_leaderboard([summary])[0]
    validate(row, schema("leaderboard"))
