# 0AD-Bench v0.1 validation

## Local validation

- Catalog JSON parsed successfully.
- All 50 expanded task specifications passed JSON Schema validation.
- All fixture references resolved against the v0.1 fixture catalog.
- Category counts: 10 economy, 10 building and technology, 10 scouting, 10 combat, 10 adaptive.
- Split counts: 45 development and 5 evaluation.
- Track counts: 50 symbolic, 50 hybrid, and 18 vision.
- Python test suite: 11 passed.
- Package wheel built successfully.

## Colab validation

- Runtime: fresh CPU Colab session.
- Installation source: `origin/0ad-bench-v0.1`.
- Catalog tasks loaded: 50.
- Smoke task: `econ-001`.
- Completed steps: 31.
- Agent decisions: 4.
- Token count: 64.
- Mean decision latency: 2 ms.
- Illegal actions: 0.
- Task success: true.
- Reference score: 84.3207.
- Trajectory files: 69.
- Agent-visible and evaluator observations were written to separate directories.
- Colab session was stopped after completion.

## Commands

```bash
PYTHONPATH=0ad-bench/src python -m pytest 0ad-bench/tests -q
PYTHONPATH=0ad-bench/src python -m zero_ad_bench.cli validate
PYTHONPATH=0ad-bench/src python -m zero_ad_bench.cli smoke --output /tmp/0ad-bench-smoke
python -m pip wheel --no-deps ./0ad-bench
```
