# exp-rts-001-thousand-turn-match

## Objective

- Start Gemma and Petra from the same fresh Spartan scenario opening.
- Give Gemma up to 1,000 reinforcement-learning turns.
- Request a Gemma strategic decision every 25 turns.
- Compare the final economy, army, structures, population, and resources.
- Record the complete native 0 A.D. replay.
- Fit the complete capture within 45 seconds.

## Configuration

| Setting | Value |
|---|---|
| Experiment ID | `exp-rts-001-thousand-turn-match` |
| Model | `google/gemma-4-E2B-it` |
| Game | 0 A.D. Release 28 |
| Gemma civilization | Sparta |
| Opponent | PetraBot, difficulty 3 |
| RL turn limit | 1,000 |
| Decision interval | 25 turns |
| Seed | 1001 |
| Result rule | Engine conquest result, then documented turn-limit score |

## Control and scoring

```python
for turn in range(1, 1001):
    if turn == 1 or (turn - 1) % 25 == 0:
        action, raw = commander.choose(observation(state), legal_actions(state))
        commands = engine_commands(state, action)
    state = game.step(commands if turn == 1 or (turn - 1) % 25 == 0 else [])

outcome = compare_players(state)
```

- An engine conquest result has priority.
- The turn-limit score is used when both sides remain active.
- Score formula: `population + 3*workers + 6*military + 8*structures + resources/100`.
- The turn-limit score is an experiment metric rather than an official 0 A.D. victory condition.

## Result

- Status: pending Colab execution.
- Completed turns: pending.
- Winner: pending.
- Result type: pending.
- Explanation: pending.

## Final statistics

| Metric | Gemma | Petra |
|---|---:|---:|
| Game state | pending | pending |
| Phase | pending | pending |
| Population | pending | pending |
| Population limit | pending | pending |
| Economic workers | pending | pending |
| Military units | pending | pending |
| Structures | pending | pending |
| Food | pending | pending |
| Wood | pending | pending |
| Stone | pending | pending |
| Metal | pending | pending |
| Total banked resources | pending | pending |
| Turn-limit score | pending | pending |

## Artifacts

- [Notebook](../notebooks/exp_rts_001_thousand_turn_match.ipynb)
- Gameplay video: pending execution.
- Native replay archive: pending execution.
- Weights & Biases run: pending execution.
