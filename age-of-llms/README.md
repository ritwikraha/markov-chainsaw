# Age of LLMs

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ritwikraha/markov-chainsaw/blob/age-of-llms/age-of-llms/notebooks/age_of_empires_with_gemma.ipynb)

[![Open exp-rts-001 in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ritwikraha/markov-chainsaw/blob/age-of-llms/age-of-llms/notebooks/exp_rts_001_thousand_turn_match.ipynb)

## Experiment index

- IDs use `exp-rts-NNN-description`.
- `NNN` defines the run order.
- Each experiment has one Markdown report in [`experiments/`](./experiments/).
- [`exp-rts-001-thousand-turn-match`](./experiments/exp-rts-001-thousand-turn-match.md): fresh Spartan start, 1,000 RL turns, final player comparison, complete native replay capture.

## Result

- Connected `google/gemma-4-E2B-it` to the real 0 A.D. Release 28 Pyrogenesis engine.
- Used the official reinforcement learning interface to read game state and submit commands.
- Converted model output into a validated JSON macro action.
- Checked action cost, population space, unit availability, and target availability.
- Played a Spartan match against PetraBot for 30 model decisions.
- Completed the verified run with zero parser fallbacks.
- Saved the authoritative 0 A.D. replay.
- Recorded a 45-second, 1280 by 720 video from the native visual replay renderer.
- Logged game metrics and artifacts to [Weights & Biases](https://wandb.ai/ritwik/gemma4-zero-ad/runs/344geg0c).

## Artifacts

- [Colab notebook](./notebooks/age_of_empires_with_gemma.ipynb)
- [Native gameplay video](./videos/age_of_empires_with_gemma_gameplay.mp4)
- [0 A.D. Release 28 replay](./replays/age_of_empires_with_gemma_release28_replay.zip)

## Reusable utilities

- [`utils/artifacts.py`](./utils/artifacts.py): checksums and replay ZIP creation.
- [`utils/experiment_stats.py`](./utils/experiment_stats.py): final player table, turn-limit score, winner selection, result explanation.
- [`utils/replay_video.py`](./utils/replay_video.py): full native replay capture and 45-second duration fitting.

```python
from utils import capture_complete_replay, compare_players

outcome = compare_players(state)
capture = capture_complete_replay(APP_RUN, replay_commands, VIDEO_PATH)
```

## Control loop

- Gemma receives economy, population, army, enemy, and structure state.
- Gemma selects one action from the legal action list.
- The adapter converts the selection into an official `zero_ad.actions` command.
- Pyrogenesis advances the simulation and returns the next state.

```python
observation_data = observation(state)
legal = legal_actions(state)
action, raw_output = commander.choose(observation_data, legal)
commands = engine_commands(state, action)
state = game.step(commands)
```

## Native replay capture

- Pyrogenesis writes a native `commands.txt` replay during the match.
- The notebook launches `-replay-visual` under Xvfb.
- FFmpeg records the real game window with terrain, units, buildings, minimap, and observer UI.
- The camera switches between the normal view and bird's-eye view during recording.

```python
visual_engine = subprocess.Popen([
    str(APP_RUN),
    f"-replay-visual={replay_commands}",
    "-xres=1280",
    "-yres=720",
])

recorder = subprocess.Popen([
    "ffmpeg", "-f", "x11grab", "-i", f"{display_name}.0",
    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(VIDEO_PATH),
])
```

## Run from this PR

- Check out the branch.

```bash
git fetch origin age-of-llms
git checkout age-of-llms
```

- Open the notebook with the Colab badge above.
- Select a T4 GPU runtime or a larger GPU runtime.
- Accept the gated Gemma 4 model terms on Hugging Face.
- Add `HF_WRITE_ACCESS`, `WANDB_KEY`, and `GITHUB_TOKEN` to Colab Secrets.
- Grant the notebook access to each secret.
- Run all cells.
- Download the generated MP4 and replay ZIP from the final cells.
