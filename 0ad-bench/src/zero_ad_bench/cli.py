"""Command-line interface for catalog and result operations."""

from __future__ import annotations

import argparse
import json
from importlib import resources
from pathlib import Path

from .leaderboard import aggregate_leaderboard, leaderboard_markdown
from .models import EpisodeMetrics
from .registry import TaskRegistry
from .scoring import score_episode


def _registry(path: str = "") -> TaskRegistry:
    return TaskRegistry.from_path(Path(path)) if path else TaskRegistry.default()


def _validate_schemas(registry: TaskRegistry) -> None:
    try:
        import jsonschema
    except ImportError as exc:
        raise SystemExit("Install zero-ad-bench[validation] to run JSON Schema validation") from exc
    schema_root = resources.files("zero_ad_bench").joinpath("schemas")
    with schema_root.joinpath("task.schema.json").open(encoding="utf-8") as stream:
        task_schema = json.load(stream)
    fixture_resource = resources.files("zero_ad_bench").joinpath("data/fixtures_v0_1.json")
    with fixture_resource.open(encoding="utf-8") as stream:
        fixture_ids = {fixture["id"] for fixture in json.load(stream)["fixtures"]}
    validator_class = getattr(jsonschema, "Draft202012Validator", jsonschema.Draft7Validator)
    validator = validator_class(task_schema)
    for task in registry.list():
        validator.validate(task.to_dict())
        fixture = task.scenario["fixture"]
        if fixture not in fixture_ids:
            raise ValueError(f"Task {task.id} references unknown fixture {fixture}")


def command_validate(args: argparse.Namespace) -> None:
    registry = _registry(args.catalog)
    _validate_schemas(registry)
    print(json.dumps(registry.summary(), indent=2, sort_keys=True))


def command_list(args: argparse.Namespace) -> None:
    for task in _registry(args.catalog).list(args.category, args.track, args.split):
        print(f"{task.id}\t{task.category}\t{','.join(task.tracks)}\t{task.title}")


def command_show(args: argparse.Namespace) -> None:
    print(json.dumps(_registry(args.catalog).get(args.task_id).to_dict(), indent=2, sort_keys=True))


def command_score(args: argparse.Namespace) -> None:
    task = _registry(args.catalog).get(args.task_id)
    value = json.loads(Path(args.metrics).read_text(encoding="utf-8"))
    allowed = EpisodeMetrics.__dataclass_fields__.keys()
    metrics = EpisodeMetrics(**{key: value[key] for key in allowed if key in value})
    print(json.dumps(score_episode(metrics, task.budget).to_dict(), indent=2, sort_keys=True))


def command_leaderboard(args: argparse.Namespace) -> None:
    summaries = [json.loads(path.read_text(encoding="utf-8")) for path in Path(args.results).rglob("summary.json")]
    rows = aggregate_leaderboard(summaries)
    if args.format == "json":
        print(json.dumps(rows, indent=2, sort_keys=True))
    else:
        print(leaderboard_markdown(rows))


def command_smoke(args: argparse.Namespace) -> None:
    from .runner import BenchmarkRunner
    from .smoke import EconomySmokeEnvironment, SmokeAgent

    task = _registry(args.catalog).get("econ-001")
    summary = BenchmarkRunner(Path(args.output)).run(
        task,
        SmokeAgent(),
        EconomySmokeEnvironment(),
        episode_id="smoke-econ-001",
    )
    print(json.dumps({"episode_id": summary["episode_id"], "success": summary["success"], "score": summary["score"]}, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="0ad-bench")
    parser.add_argument("--catalog", default="", help="Optional task catalog JSON")
    commands = parser.add_subparsers(dest="command", required=True)

    validate = commands.add_parser("validate", help="Validate the v0.1 catalog")
    validate.set_defaults(function=command_validate)

    listing = commands.add_parser("list", help="List benchmark tasks")
    listing.add_argument("--category", choices=["economy", "building-tech", "scouting", "combat", "adaptive"])
    listing.add_argument("--track", choices=["symbolic", "vision", "hybrid"])
    listing.add_argument("--split", choices=["development", "evaluation"])
    listing.set_defaults(function=command_list)

    show = commands.add_parser("show", help="Show one task")
    show.add_argument("task_id")
    show.set_defaults(function=command_show)

    score = commands.add_parser("score", help="Score an episode metrics JSON")
    score.add_argument("task_id")
    score.add_argument("metrics")
    score.set_defaults(function=command_score)

    leaderboard = commands.add_parser("leaderboard", help="Aggregate episode summaries")
    leaderboard.add_argument("results")
    leaderboard.add_argument("--format", choices=["markdown", "json"], default="markdown")
    leaderboard.set_defaults(function=command_leaderboard)

    smoke = commands.add_parser("smoke", help="Run an in-process harness smoke episode")
    smoke.add_argument("--output", default="results/smoke")
    smoke.set_defaults(function=command_smoke)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
