from zero_ad_bench import aggregate_leaderboard, leaderboard_markdown


def summary(agent, score, success):
    return {
        "agent": agent,
        "track": "symbolic",
        "category": "economy",
        "success": success,
        "score": {"total": score},
        "metrics": {
            "action_count": 10,
            "total_tokens": 100,
            "mean_latency_ms": 20,
            "illegal_action_rate": 0,
        },
    }


def test_leaderboard_orders_by_mean_score():
    rows = aggregate_leaderboard([
        summary("agent-b", 50, False),
        summary("agent-a", 80, True),
        summary("agent-a", 60, False),
    ])
    assert [row["agent"] for row in rows] == ["agent-a", "agent-b"]
    assert rows[0]["mean_score"] == 70
    assert "| 1 | symbolic | agent-a |" in leaderboard_markdown(rows)
