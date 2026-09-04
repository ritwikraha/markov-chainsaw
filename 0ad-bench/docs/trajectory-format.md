# Trajectory format

## metadata.json

Contains benchmark version, episode ID, complete task specification, agent name, and seeds.

## observations/

Contains one JSON observation for step 0 and every environment step. Filenames use six-digit zero padding.

## actions.jsonl

Contains each requested agent action and its legality result. Simulation-only `wait` steps between decisions are omitted from action-cost accounting.

## evaluator_observations/

Contains privileged structured state used for goal and reward calculation. This directory is withheld from vision-track agents during execution.

## reasoning.jsonl

Contains optional concise decision rationales. An empty rationale is valid. Hidden reasoning traces are outside the required record.

## rewards.jsonl

Contains per-step environment reward and normalized component information.

## replay/

Contains the authoritative native 0 A.D. replay or an adapter-specific equivalent for smoke environments.

## summary.json

Contains final observation, success, raw metrics, complete score breakdown, task ID, category, track, agent, and seed.
