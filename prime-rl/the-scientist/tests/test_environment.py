import asyncio
import json

import pytest
from verifiers.types import AssistantMessage, UserMessage

from the_scientist.environment import (
    ScientistEnv,
    canonical_equation,
    decode_polynomial,
    evaluate_polynomial,
    generate_universe,
    hidden_probe_score,
    load_environment,
    parse_polynomial,
    parse_text_experiment,
    structure_score,
)


def test_parser_accepts_equivalent_polynomials():
    assert parse_polynomial("2*x1 + x2**2 - x1*x2 + 3") == parse_polynomial(
        "3 + x2*x2 + x1*2 - x2*x1"
    )


@pytest.mark.parametrize(
    "expression",
    ["__import__('os').system('id')", "x1 / x2", "x1**3", "(1).__class__", "float('nan')"],
)
def test_parser_rejects_unsafe_or_unsupported_syntax(expression):
    with pytest.raises((ValueError, SyntaxError)):
        parse_polynomial(expression)


def test_text_experiment_parser_accepts_one_strict_command():
    assert parse_text_experiment("EXPERIMENT x1=-2 x2=3") == (-2, 3)
    assert parse_text_experiment("thinking\n  experiment x1 = +1 x2 = 0  \n") == (1, 0)


@pytest.mark.parametrize(
    "content",
    [
        "EXPERIMENT x1=1",
        "EXPERIMENT x1=1 x2=2 extra=true",
        "EXPERIMENT x1=1.5 x2=2",
        {"x1": 1, "x2": 2},
    ],
)
def test_text_experiment_parser_rejects_malformed_commands(content):
    assert parse_text_experiment(content) is None


def test_text_experiment_parser_uses_only_first_complete_command():
    content = "EXPERIMENT x1=1 x2=2\nEXPERIMENT x1=2 x2=1"
    assert parse_text_experiment(content) == (1, 2)


def test_universe_is_reproducible_and_canonical():
    first = generate_universe(level=3, seed=42, budget=6, domain=(-3, 3))
    second = generate_universe(level=3, seed=42, budget=6, domain=(-3, 3))
    assert first == second
    polynomial = decode_polynomial(first["coefficients"])
    assert parse_polynomial(canonical_equation(polynomial)) == polynomial


def test_environment_has_disjoint_splits_and_no_prompt_leakage():
    env = load_environment(level=2, num_train=3, num_eval=3, seed=5)
    train = env.get_dataset()
    evaluation = env.get_eval_dataset()
    assert set(train["answer"]).isdisjoint(evaluation["answer"])
    for prompt, answer in zip(train["prompt"], train["answer"], strict=True):
        truth = json.loads(answer)
        assert truth["equation"] not in json.dumps(prompt)


def test_laboratory_enforces_budget_and_records_duplicates():
    universe = generate_universe(level=1, seed=9, budget=2, domain=(-3, 3))
    lab_state = {
        "polynomial": decode_polynomial(universe["coefficients"]),
        "remaining": 2,
        "domain": (-3, 3),
        "observations": [],
    }
    env = ScientistEnv(budget=2, domain=(-3, 3), dataset=load_environment(num_train=1).get_dataset())
    first = json.loads(env.experiment(1, 0, lab_state))
    second = json.loads(env.experiment(1, 0, lab_state))
    exhausted = json.loads(env.experiment(0, 0, lab_state))
    assert first["duplicate"] is False
    assert second["duplicate"] is True
    assert exhausted["error"] == "experiment budget exhausted"
    assert lab_state["remaining"] == 0


def test_text_protocol_executes_experiment_without_native_tool_schema():
    env = load_environment(level=1, num_train=1, num_eval=1, seed=9, budget=2, protocol="text")
    universe = generate_universe(level=1, seed=9, budget=2, domain=(-3, 3))
    assistant = AssistantMessage(role="assistant", content="EXPERIMENT x1=1 x2=0")
    state = {
        "answer": json.dumps(universe),
        "trajectory": [{"completion": [assistant]}],
    }
    asyncio.run(env.setup_state(state))

    assert env.tool_defs == []
    assert env.sampling_args["stop"] == ["\n"]
    assert asyncio.run(env.no_tools_called(state)) is False
    response = asyncio.run(env.env_response([assistant], state))

    assert isinstance(response[0], UserMessage)
    assert '"experiments_remaining": 1' in response[0].content
    assert state["text_experiment_calls"] == 1
    assert len(state["lab_state"]["observations"]) == 1


def test_text_protocol_stops_for_final_json():
    env = load_environment(num_train=1, num_eval=1, protocol="text")
    assistant = AssistantMessage(
        role="assistant",
        content='{"family":"affine","equation":"x1+x2","confidence":0.5}',
    )
    state = {"trajectory": [{"completion": [assistant]}]}
    assert asyncio.run(env.no_tools_called(state)) is True


def test_native_protocol_keeps_tool_schema():
    env = load_environment(num_train=1, num_eval=1, protocol="native")
    assert env.tool_defs is not None
    assert [tool.name for tool in env.tool_defs] == ["experiment"]


def test_environment_rejects_unknown_protocol():
    with pytest.raises(ValueError, match="protocol must be one of"):
        load_environment(num_train=1, num_eval=1, protocol="carrier-pigeon")


def test_invalid_experiment_consumes_budget():
    universe = generate_universe(level=1, seed=9, budget=1, domain=(-3, 3))
    lab_state = {
        "polynomial": decode_polynomial(universe["coefficients"]),
        "remaining": 1,
        "domain": (-3, 3),
        "observations": [],
    }
    env = ScientistEnv(budget=1, domain=(-3, 3), dataset=load_environment(num_train=1).get_dataset())
    response = json.loads(env.experiment(99, 0, lab_state))
    assert "error" in response
    assert lab_state["remaining"] == 0


def test_correct_law_gets_full_reward_and_wrong_law_does_not():
    universe = generate_universe(level=3, seed=21, budget=6, domain=(-3, 3))
    completion = [
        {
            "role": "assistant",
            "content": json.dumps(
                {"family": universe["family"], "equation": universe["equation"], "confidence": 1.0}
            ),
        }
    ]
    wrong_completion = [
        {
            "role": "assistant",
            "content": json.dumps({"family": universe["family"], "equation": "0", "confidence": 0.1}),
        }
    ]
    state = {"answer": json.dumps(universe)}
    assert asyncio.run(hidden_probe_score(completion, state)) == pytest.approx(1.0)
    assert asyncio.run(structure_score(completion, state)) == pytest.approx(1.0)
    assert asyncio.run(hidden_probe_score(wrong_completion, state)) < 1.0


def test_polynomial_evaluation_matches_known_law():
    polynomial = parse_polynomial("2*x1 + x2**2")
    assert evaluate_polynomial(polynomial, 1, 0) == 2
    assert evaluate_polynomial(polynomial, 2, 0) == 4
    assert evaluate_polynomial(polynomial, 0, 2) == 4
