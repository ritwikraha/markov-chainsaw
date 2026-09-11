from zero_ad_bench.predicates import evaluate_predicate, task_reward


OBSERVATIONS = [
    {"player": {"resources": {"wood": 100}, "state": "active"}},
    {"player": {"resources": {"wood": 250}, "state": "active"}},
    {"player": {"resources": {"wood": 175}, "state": "active"}},
]


def test_final_and_max_scopes():
    assert not evaluate_predicate({"path": "player.resources.wood", "op": "gte", "value": 200}, OBSERVATIONS)
    assert evaluate_predicate({"path": "player.resources.wood", "op": "gte", "value": 200, "scope": "max"}, OBSERVATIONS)


def test_fraction_goal_returns_partial_credit():
    goal = {
        "aggregation": "fraction",
        "predicates": [
            {"path": "player.state", "op": "eq", "value": "active"},
            {"path": "player.resources.wood", "op": "gte", "value": 200},
        ],
    }
    assert task_reward(goal, OBSERVATIONS) == 0.5
