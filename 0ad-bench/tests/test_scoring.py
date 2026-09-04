from zero_ad_bench import EpisodeMetrics, score_episode


def test_reference_score_breakdown():
    metrics = EpisodeMetrics(
        task_reward=1.0,
        economy_reward=1.0,
        military_reward=0.5,
        survival_reward=1.0,
        action_count=10,
        illegal_action_count=1,
        prompt_tokens=400,
        completion_tokens=100,
        latency_ms_total=5000,
    )
    budget = {"max_actions": 20, "max_agent_actions": 20, "max_tokens": 1000, "max_mean_latency_ms": 1000}
    score = score_episode(metrics, budget)
    assert score.total == 84.5
    assert score.illegal_action_penalty == 0.5
