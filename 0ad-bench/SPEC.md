# 0AD-Bench v0.1 specification

## 1. Purpose

0AD-Bench measures autonomous-agent capability in a deterministic real-time strategy environment. The benchmark separates task achievement, economy, military performance, survival, action efficiency, token use, latency, legality, perception, planning, control, and recovery.

Full-game win rate is one benchmark tier. Shorter capability tasks provide denser and more diagnostic measurements.

## 2. Versioning

- Benchmark version: `0.1`.
- Engine compatibility target: 0 A.D. Release 28.x.
- Schema version: `0.1`.
- Catalog changes that alter goals, fixtures, budgets, or scoring require a new benchmark version.
- Documentation and spelling fixes can retain the current version when they preserve episode behavior.

## 3. Tracks

### 3.1 Symbolic

- Input: observations conforming to `observation.schema.json`.
- Output: actions conforming to `action.schema.json`.
- Intended use: open models, reinforcement learning, ablations, and high-volume evaluation.

### 3.2 Vision

- Input: native rendered game frames and task instructions.
- Output: mouse and keyboard events through a track adapter.
- Structured game telemetry is limited to episode control fields such as step and remaining budget.
- Intended use: perception and computer-control evaluation.

### 3.3 Hybrid

- Input: native rendered frames plus selected symbolic telemetry.
- Output: versioned macro actions.
- Intended use: reasoning and planning with practical software interfaces.

Scores from different tracks occupy separate leaderboard columns and rankings.

## 4. Task catalog

The v0.1 catalog contains 50 tasks:

- `econ-001` through `econ-010`.
- `tech-001` through `tech-010`.
- `scout-001` through `scout-010`.
- `combat-001` through `combat-010`.
- `adapt-001` through `adapt-010`.

Tasks ending in `010` form the evaluation split. The remaining 45 tasks form the development split.

Each task defines:

- A stable identifier and version.
- One category and one split.
- Supported tracks.
- A deterministic fixture, map, engine seed, civilization, and opponent.
- Step, action, token, and latency budgets.
- Declarative goal predicates.
- Zero or more deterministic perturbations.
- Analysis tags.

## 5. Determinism

An official episode fixes:

- 0 A.D. release and mod checksums.
- Task catalog version.
- Fixture version.
- Map and AI seed.
- Player civilization and opponent configuration.
- Simulation turn length.
- Agent sampling seed and decoding parameters.
- Decision interval.
- Perturbation step and parameters.
- Observation and action schema versions.

The native 0 A.D. replay command stream is the authoritative game record. Repeated deterministic agents must produce the same action stream and final state under the same configuration.

## 6. Episode lifecycle

1. Resolve the task and fixture.
2. Start a clean engine process.
3. Reset the scenario with the task seed and replay saving enabled.
4. Emit observation step 0.
5. Request an agent decision at step 1 and each decision interval.
6. Validate the action against the schema and current legal-action set.
7. Replace an invalid action with `wait` and increment the illegal-action count.
8. Apply scheduled perturbations before the decision at their specified step.
9. Advance one RL step.
10. Record agent-visible observation, evaluator observation, action, rationale, reward, token count, and latency.
11. Stop on engine termination or the task step limit.
12. Evaluate goal predicates and compute the reference score.
13. Flush the native replay and write `summary.json`.
14. Close the engine process.

## 7. Observation contract

Symbolic observations contain:

- Episode and time identifiers.
- Track.
- Player and opponent state.
- Resources and gatherer allocation.
- Population, units, structures, phase, and benchmark statistics.
- Visible entities with stable IDs, owner, template, and position.
- Map telemetry required by the active task.
- Benchmark event flags.
- Raw engine event log when available.

Fog-of-war tasks expose only currently visible or previously observed opponent information. Fixture-specific telemetry must be declared and applied consistently to every evaluated agent.

Vision-track agents receive the rendered frame plus episode control fields. Structured evaluator observations are written to a separate stream and withheld from the agent during execution.

## 8. Action contract

The v0.1 macro-action vocabulary is:

```text
wait gather train construct research walk scout attack defend retreat
```

Actions can reference controlled entity IDs, a target entity, a template, a count, or a map position. The adapter validates entity ownership, availability, affordability, target visibility, population capacity, and placement before submitting official engine commands.

Vision-track adapters record each mouse and keyboard event and expose the same action-cost counters.

Macro actions use `action.schema.json`. Vision control events use `vision-action.schema.json`.

## 9. Goal evaluation

Goals contain one or more predicates with:

- A dotted observation path.
- An operator: `eq`, `ne`, `gte`, `lte`, `gt`, `lt`, or `contains`.
- An expected value.
- A scope: final value, any observed value, maximum, or minimum.

Goal aggregation modes:

- `all`: full credit when every predicate passes; otherwise fractional diagnostic credit.
- `any`: full credit when at least one predicate passes.
- `fraction`: fraction of predicates passed.

An episode reports task success only when task reward equals `1.0`.

## 10. Scoring

All normalized rewards and costs lie in `[0, 1]`.

$$
S = \max(0,
50R_{task}
+20R_{economy}
+15R_{military}
+15R_{survival}
-5C_{actions}
-5C_{tokens}
-5C_{latency}
-5C_{illegal})
$$

Definitions:

- `R_task`: declarative goal reward.
- `R_economy`: task fixture economy reward normalized against its reference target.
- `R_military`: task fixture combat reward normalized against its reference target.
- `R_survival`: survival and protected-objective reward.
- `C_actions`: agent decisions divided by the task action budget.
- `C_tokens`: prompt plus completion tokens divided by the token budget.
- `C_latency`: mean decision latency divided by the latency budget.
- `C_illegal`: illegal actions divided by all agent decisions.

The final episode score is between 0 and 100. The full component breakdown is required in `summary.json`.

## 11. Perturbation protocol

Supported v0.1 perturbations:

- Remove controlled units.
- Remove a resource cluster.
- Spawn a timed attack.
- Change enemy composition.
- Destroy a controlled structure.
- Block a route.
- Apply a stored-resource shock.

Each perturbation has a fixed step and parameters. The benchmark event stream records application and recovery. Adaptive tasks measure whether the agent observes the change, stops an obsolete plan, selects a new plan, and restores task progress.

## 12. Evaluation protocol

- Run every supported task in the selected track.
- Use three agent sampling seeds per task: 0, 1, and 2.
- Keep the task engine seed fixed.
- Reset agent memory at the start of each episode unless the submission declares cross-episode learning.
- Record failures and timeouts as completed episodes with their earned score.
- Report development and evaluation splits separately.
- Aggregate episode scores within each category first.
- Compute overall score as the arithmetic mean of available category means.
- Report success rate, action count, token count, latency, and illegal-action rate beside score.
- Keep track leaderboards separate.

## 13. Submission metadata

Every leaderboard submission declares:

- Agent name and version.
- Model identifier, parameter count, and weight availability.
- Observation track.
- Memory method.
- Planning method.
- Action granularity.
- Context-window size.
- Decoding and sampling configuration.
- Hardware.
- Maximum and average test-time compute.
- Demonstrations, fine-tuning, reinforcement learning, or other training used.
- Repository revision and container digest.

## 14. Trajectory contract

Each episode directory contains:

```text
metadata.json
observations/000000.json ...
evaluator_observations/000000.json ...
actions.jsonl
reasoning.jsonl
rewards.jsonl
replay/
summary.json
```

Decision rationale is optional and limited to concise agent-provided text. Evaluation must remain valid when rationale storage is disabled.

## 15. Result validity

An official result requires:

- Successful schema validation.
- Exact engine and benchmark versions.
- Complete task coverage for the reported split and track.
- Three agent seeds per task.
- Native replays for engine-backed episodes.
- Complete metric and score components.
- Submission metadata.
- Public or auditor-accessible episode summaries.

## 16. Future versions

Planned extensions include procedural unseen maps, unseen civilizations, adaptive opponents, native vision-control fixtures, human baselines, model scaling studies, offline trajectory releases, and learning tracks.
