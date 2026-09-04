# 0AD-Bench

0AD-Bench evaluates long-horizon autonomous agents inside 0 A.D. Release 28.

Version 0.1 provides:

- 50 deterministic task specifications.
- Five categories with ten tasks each.
- Symbolic, vision, and hybrid tracks.
- Versioned observation, macro-action, vision-action, task, summary, and leaderboard schemas.
- A reference closed-loop runner.
- Token, action, latency, and illegal-action penalties.
- Perturbation hooks for recovery and replanning tests.
- Complete trajectory recording.
- A macro-action adapter for the official `zero_ad` RL client.
- A Colab notebook for catalog inspection, scoring, and smoke evaluation.

The package and notebook validation record is available in [VALIDATION.md](./VALIDATION.md).

[![Open 0AD-Bench v0.1 in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ritwikraha/markov-chainsaw/blob/0ad-bench-v0.1/0ad-bench/notebooks/0ad_bench_v0_1.ipynb)

## Task suite

| Category | Tasks | Primary capability |
|---|---:|---|
| Economy | 10 | Resource allocation and population growth |
| Building and technology | 10 | Construction, production, and phase progression |
| Scouting | 10 | Exploration, detection, and opponent observation |
| Combat | 10 | Micro-control, defense, retreat, and conquest |
| Adaptive | 10 | Closed-loop replanning after controlled perturbations |

## Tracks

- `symbolic`: structured state and structured macro actions.
- `vision`: rendered frames and mouse or keyboard actions supplied by a vision adapter.
- `hybrid`: rendered frames plus selected structured telemetry and macro actions.

## Quick start

```bash
cd 0ad-bench
python -m pip install -e ".[test]"
0ad-bench validate
0ad-bench list --category adaptive
pytest
```

Expected catalog summary:

```json
{
  "task_count": 50,
  "categories": {
    "adaptive": 10,
    "building-tech": 10,
    "combat": 10,
    "economy": 10,
    "scouting": 10
  },
  "splits": {"development": 45, "evaluation": 5},
  "tracks": {"hybrid": 50, "symbolic": 50, "vision": 18}
}
```

## Run one episode

```python
from pathlib import Path
from zero_ad_bench import BenchmarkRunner, TaskRegistry

registry = TaskRegistry.default()
task = registry.get("econ-001")
runner = BenchmarkRunner(Path("results"))
summary = runner.run(task, agent, environment)
print(summary["score"]["total"])
```

`agent` implements `act(observation, legal_actions, context)` and returns an `AgentDecision`. `environment` implements the runner protocol or uses `ZeroADEnvironment` from `zero_ad_bench.adapters`.

## Repository structure

```text
0ad-bench/
├── SPEC.md
├── LEADERBOARD.md
├── notebooks/
├── src/zero_ad_bench/
│   ├── adapters/
│   ├── baselines/
│   ├── data/
│   ├── schemas/
│   ├── runner.py
│   ├── scoring.py
│   └── trajectory.py
└── tests/
```

## Episode artifacts

```text
episode_00421/
├── metadata.json
├── observations/
├── evaluator_observations/
├── actions.jsonl
├── reasoning.jsonl
├── rewards.jsonl
├── replay/
└── summary.json
```

`reasoning.jsonl` stores a concise decision rationale supplied by the evaluated agent. Hidden reasoning traces are outside the benchmark contract.

Vision episodes keep structured scoring telemetry in `evaluator_observations/`. That directory is withheld from the agent during execution.

## Current implementation boundary

- The benchmark package, schemas, catalog, runner, scoring, aggregation, and official RL-client adapter are executable.
- Fixture descriptors define deterministic engine setup requirements for all 50 tasks.
- Engine-side scenario and trigger realization for every fixture is the next milestone.
- The existing Age of LLMs notebooks demonstrate the pinned Release 28 engine, real RL endpoint, native replay generation, and video capture used by this benchmark.

Read [SPEC.md](./SPEC.md) for the normative v0.1 protocol and [docs/engine-integration.md](./docs/engine-integration.md) for the 0 A.D. boundary.
