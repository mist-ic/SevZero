# SevZero expert trajectories (SFT)

## Sources

- Synthetic expert rollouts from frontier models (Gemini 3.1 Pro, Azure OpenAI, Azure AI Inference)
  against the local OpenEnv `server.app` SevZero environment.

## Filtering

- Episodes with final grader `score` **≥** `0.75` are included.

## Schema

- Each example has a `messages` list (Llama-3.1-8B-Instruct–style SFT) and `meta` (episode / step provenance):
  - `system`: SRE on-call system prompt (same as `inference.SYSTEM_PROMPT` in the repo)
  - `user`: JSON-serialized observation (shrink to ≤ 2048 tokens for the user part)
  - `assistant`: one JSON object `{"action_type": "...", "params": {...}}`

## Stats (from `build_stats.json` at publish time)

{
  "episodes_total_seen": 90,
  "episodes_kept": 42,
  "episodes_dropped": 48,
  "mean_episode_score_kept": 0.836021,
  "train_rows": 853,
  "eval_rows": 80,
  "max_prompt_token_length": 2,
  "max_observation_user_token_budget": 2048,
  "min_score_filter": 0.75
}

## Parquet

- Splits `train` and `eval` are also pushed in Parquet for fast `datasets.load_dataset`.
