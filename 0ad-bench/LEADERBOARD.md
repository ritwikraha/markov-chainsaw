# 0AD-Bench leaderboard protocol

## Ranking

- Maintain separate rankings for symbolic, vision, and hybrid tracks.
- Rank by overall mean score.
- Overall mean score is the arithmetic mean of category means.
- Use evaluation-split score for the primary ranking.
- Use development-split score for iteration and ablation reporting.
- Break ties by task success rate, then lower token count, then lower action count, then lower mean latency.

## Required table

| Rank | Track | Agent | Score | Success | Economy | Building and tech | Scouting | Combat | Adaptive | Actions | Tokens | Latency ms | Illegal rate |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | symbolic | example-agent | 72.50 | 66.7% | 80.0 | 75.0 | 70.0 | 68.0 | 69.5 | 31.0 | 12000 | 450 | 0.2% |

## Submission record

```json
{
  "agent": "example-agent",
  "agent_version": "1.0.0",
  "model": "model-identifier",
  "track": "symbolic",
  "memory": "sliding-window",
  "planning": "hierarchical",
  "action_granularity": "macro",
  "hardware": "NVIDIA L4",
  "benchmark_version": "0.1",
  "engine_version": "0.28.0",
  "repository_revision": "git-sha",
  "container_digest": "sha256:..."
}
```

Run:

```bash
0ad-bench leaderboard results/ --format markdown
```
