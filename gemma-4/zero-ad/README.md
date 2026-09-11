# Age of Empires with Gemma

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ritwikraha/markov-chainsaw/blob/main/gemma-4/zero-ad/gemma4_zero_ad_colab.ipynb)

This experiment connects `google/gemma-4-E2B-it` to the real [0 A.D.](https://play0ad.com/) Release 28 simulation. Gemma observes a compact view of the live economy, population, army, enemy, and map; chooses one legal macro action as JSON; and the official 0 A.D. reinforcement-learning interface executes it against PetraBot.

This runs the actual open-source Pyrogenesis game engine in Colab. Gemma plays through the official RL interface. After the match, the notebook launches 0 A.D.'s native visual replay mode inside a virtual display and records it with FFmpeg. The video contains the real terrain, units, buildings, minimap, observer interface, and camera view.

## Run it

1. Open [`gemma4_zero_ad_colab.ipynb`](./gemma4_zero_ad_colab.ipynb) in Colab.
2. Select a GPU runtime (T4 or better).
3. Accept access to [`google/gemma-4-E2B-it`](https://huggingface.co/google/gemma-4-E2B-it).
4. Grant the notebook access to the private Colab secrets `HF_WRITE_ACCESS`, `WANDB_KEY`, and `GITHUB_TOKEN`.
5. Run all cells. The first setup downloads the official 0 A.D. 0.28.0 AppImage (about 1.7 GB) and Gemma weights.

At the end, the playback cell embeds and downloads `age_of_empires_with_gemma_gameplay.mp4`. The export cell also produces a native replay archive. Extract it into 0 A.D. Release 28's replay directory to watch the match with the desktop renderer. Metrics and both artifacts are logged to the `gemma4-zero-ad` Weights & Biases project.

## Latest authenticated run

- [Native gameplay video](./outputs/age_of_empires_with_gemma_gameplay.mp4)
- [Executed notebook with outputs](./outputs/age_of_empires_with_gemma_output.ipynb)
- [Native Release 28 replay](./outputs/gemma4_zero_ad_release28_replay.zip)
- [Weights & Biases run](https://wandb.ai/ritwik/gemma4-zero-ad/runs/344geg0c)

This run used Gemma 4 for all 30 decisions with zero parser fallbacks. The 45-second, 1280 by 720 MP4 was recorded directly from 0 A.D.'s visual replay mode. The ZIP is the authoritative simulation replay for desktop playback.

## Architecture

```text
0 A.D. Release 28 state -> compact observation -> Gemma 4
        ^                                         |
        |                                         v
 official RL interface <- validated macro action JSON
```

The notebook pins the official Release 28 AppImage and the matching `zero_ad` client source. It launches Pyrogenesis as an unprivileged local user, binds the RL HTTP service to `127.0.0.1`, validates model output, rate-limits decisions, saves the native replay, and always terminates the engine process.

This is an offline research demo. It does not connect to the multiplayer lobby and should not be adapted for competitive automation.
