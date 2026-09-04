# OSS model comparison v0.1

## Result

| Rank | Model | Correct | Accuracy | Legal actions | Mean latency | Tokens |
|---:|---|---:|---:|---:|---:|---:|
| 1 | `mistralai/Ministral-3-3B-Instruct-2512-BF16` | 15/15 | 100.0% | 100.0% | 1,480 ms | 4,106 |
| 2 | `google/gemma-4-E2B-it` | 14/15 | 93.3% | 100.0% | 1,909 ms | 3,941 |
| 3 | `microsoft/Phi-4-mini-instruct` | 9/15 | 60.0% | 66.7% | 1,638 ms | 3,600 |
| 4 | `Qwen/Qwen3.5-4B` | 5/15 | 33.3% | 53.3% | 2,923 ms | 4,008 |

- Ministral 3 won this compatibility run. It selected the expected action for every probe, emitted valid macro-action JSON every time, and had the lowest mean inference latency.
- Gemma 4 missed one combat probe. It attacked a raider instead of issuing the requested civic-centre defense action. All 15 outputs satisfied the action contract.
- Phi-4 Mini had five illegal outputs caused by schema errors and one legal but incorrect retreat decision.
- Qwen3.5 had seven illegal outputs. Several contained the correct candidate inside an extra `action_candidates` wrapper, which the strict action schema rejects. Three additional outputs were legal but selected the wrong candidate.

## Method

- Hardware: NVIDIA L4 with 22.0 GiB reported VRAM.
- Runtime: PyTorch 2.11.0 with CUDA 12.8.
- Decoding: greedy, `do_sample=False`, maximum 96 generated tokens.
- Loading: BF16, one model at a time, with GPU memory released between models.
- Suite: 15 deterministic single-decision probes, with three probes in each benchmark category.
- Inputs: one structured symbolic state, one objective, legal action types, and three candidate macro actions.
- Outputs: one strict JSON macro action.
- Metrics: expected-action accuracy, legal-action rate, inference latency, and input plus output tokens.
- Run date: 2026-09-05 in Asia/Kolkata.

The selected checkpoints are documented by their official model cards: [Gemma 4 E2B Instruct](https://huggingface.co/google/gemma-4-E2B-it), [Qwen3.5 4B](https://huggingface.co/Qwen/Qwen3.5-4B), [Ministral 3 3B Instruct BF16](https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512-BF16), and [Phi-4 Mini Instruct](https://huggingface.co/microsoft/Phi-4-mini-instruct).

## Category accuracy

| Model | Economy | Building and technology | Scouting | Combat | Adaptive |
|---|---:|---:|---:|---:|---:|
| Ministral 3 3B | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| Gemma 4 E2B | 100.0% | 100.0% | 100.0% | 66.7% | 100.0% |
| Phi-4 Mini | 33.3% | 33.3% | 100.0% | 66.7% | 66.7% |
| Qwen3.5 4B | 33.3% | 33.3% | 66.7% | 33.3% | 0.0% |

## Artifacts

- `leaderboard.csv` contains the sortable summary.
- `results.json` contains every episode summary and category aggregate.
- `raw-trajectories.zip` contains 60 complete episode directories with metadata, agent-visible observations, evaluator observations, actions, reasoning status, rewards, and summaries.
- `raw-trajectories.zip` SHA-256: `a4a07c68cc2a991e620f0fb4553665de05bbf9c47d14507be7ba9c4cdc6ab691`.
- The executed notebook is `notebooks/oss_model_comparison_output.ipynb`.
- The logged comparison is available in [Weights & Biases](https://wandb.ai/ritwik/0ad-bench/runs/yen647ng).

## Reproduce

1. Open `notebooks/oss_model_comparison.ipynb` in Colab.
2. Select an L4 GPU runtime.
3. Add `HF_WRITE_ACCESS` to Colab secrets and grant notebook access.
4. Add `WANDB_KEY` if experiment logging is wanted.
5. Run every cell.
6. Download `/content/oss_model_comparison.zip` for the complete trajectories.

## Interpretation boundary

- This run verifies model integration, symbolic state use, action selection, strict JSON compliance, and trajectory capture.
- The probes are one-decision development checks with visible candidate actions.
- The sample size is 15 with one deterministic pass per model, so the ranking has no confidence interval.
- These results are separate from the official 50-task engine-backed benchmark, visual control, and full 0 A.D. matches.
- A model-specific output normalizer could raise Phi and Qwen legal-action rates. This report preserves the raw strict-contract result so the integration cost remains measurable.
