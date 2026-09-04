# 0 A.D. engine integration

## Supported boundary

`ZeroADEnvironment` wraps an official `zero_ad.ZeroAD` client connected to the Release 28 RL endpoint.

```python
from zero_ad_bench.adapters import ZeroADEnvironment

environment = ZeroADEnvironment(
    game=game,
    actions_module=zero_ad.actions,
    reset_payload_factory=build_scenario,
    perturbation_hook=apply_fixture_perturbation,
    replay_exporter=export_native_replay,
    close_hook=engine.stop,
)
```

The surrounding process launcher should use:

```text
-autostart-nonvisual
-autostart-player=-1
-rl-interface=127.0.0.1:6000
-quickstart
-nosound
```

## Scenario factory

The reset factory receives a `TaskSpec` and engine seed. It must:

- Resolve the named fixture.
- Set map, player civilization, opponent, difficulty, and behavior.
- Set `Seed` and `AISeed`.
- Enable the required benchmark trigger script.
- Return a Release 28 scenario payload.

## Perturbation hook

Adaptive tasks require a deterministic trigger implementation. The hook receives the scheduled perturbation and current engine state. It must apply the declared operation exactly once and emit a benchmark event containing task ID, step, type, parameters, and affected entity IDs.

## Telemetry

The adapter emits core state directly from the official RL response. Fixture trigger scripts add benchmark telemetry under:

```json
{
  "benchmarkMap": {},
  "benchmarkEvents": {},
  "players": [{"statistics": {}}]
}
```

Task-specific telemetry names appear in the catalog goal paths. Each fixture contract must implement every referenced field.

## Replay

- Enable `save_replay=True` on reset.
- Flush the replay before closing the engine.
- Copy the complete replay directory into the episode `replay/` directory.
- Store the Release 28 engine version and mod list in episode metadata.
- Use native `-replay-visual` for rendered verification or video generation.
