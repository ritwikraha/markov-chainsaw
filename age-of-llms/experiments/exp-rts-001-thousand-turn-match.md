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

- Status: completed on an L4 Colab runtime.
- Completed RL turns: 1,000.
- Gemma decisions: 40.
- Parser fallbacks: 0.
- Winner: Petra.
- Result type: turn-limit leader.
- Turn-limit score: Petra 626.0, Gemma 264.5.

## Why Petra led

- Both sides remained active at turn 1,000, so the experiment used the documented turn-limit score.
- Petra reached the town phase while Gemma remained in the village phase.
- Petra expanded its population limit to 70. Gemma stayed at 20 and ended at 18 population.
- Petra fielded 40 military units compared with Gemma's 15.
- Petra had 60 economic-capable units compared with Gemma's 17.
- Petra built 16 structures compared with Gemma's 12.
- Gemma finished with 950 banked resources compared with Petra's 896, but did not convert that resource advantage into population capacity, a larger army, or a phase advance.
- Petra therefore led because it converted its economy into growth and military capacity more effectively within the 1,000-turn limit.

## Final statistics

| Metric | Gemma | Petra |
|---|---:|---:|
| Game state | active | active |
| Phase | village | town |
| Population | 18 | 69 |
| Population limit | 20 | 70 |
| Economic-capable units | 17 | 60 |
| Military units | 15 | 40 |
| Structures | 12 | 16 |
| Food | 30 | 410 |
| Wood | 320 | 146 |
| Stone | 300 | 140 |
| Metal | 300 | 200 |
| Total banked resources | 950 | 896 |
| Turn-limit score | 264.5 | 626.0 |

## Video verification

- Native replay simulation: 2,999 simulation turns and 599.8 simulated seconds.
- Full renderer capture: 620.0 seconds.
- Final MP4: 45.0 seconds, H.264, 960 by 540, 24 fps.
- The last sampled frame shows 9:59 and the native replay-finished dialog.
- Video SHA-256: `23698943630645c16e227721675aa39ebac56c50f94c5273d52746fe0c44301b`.
- Replay SHA-256: `6c56b32c352fceda607bf908a40ddbe3f45c2281b09687395c9952ebdd30ee83`.

## Artifacts

- [Notebook](../notebooks/exp_rts_001_thousand_turn_match.ipynb)
- [Complete native gameplay video](../videos/exp-rts-001-thousand-turn-match.mp4)
- [0 A.D. Release 28 replay](../replays/exp-rts-001-thousand-turn-match-release28-replay.zip)
- [Weights & Biases run](https://wandb.ai/ritwik/gemma4-zero-ad/runs/4ngff5ss)
