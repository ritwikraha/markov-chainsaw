"""The Scientist: discover a hidden polynomial with a limited laboratory."""

from __future__ import annotations

import ast
import json
import math
import random
from collections.abc import Mapping
from typing import Any

import verifiers as vf
from datasets import Dataset

Exponent = tuple[int, int]
Polynomial = dict[Exponent, float]

VARIABLES = ("x1", "x2")
BASES: dict[int, tuple[Exponent, ...]] = {
    1: ((0, 0), (1, 0), (0, 1)),
    2: ((0, 0), (1, 0), (0, 1), (2, 0), (0, 2)),
    3: ((0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2)),
}
FAMILY_NAMES = {1: "affine", 2: "additive_polynomial", 3: "interaction_polynomial"}
COEFFICIENTS = (-3, -2, -1, 1, 2, 3)


def _clean(poly: Polynomial) -> Polynomial:
    return {power: value for power, value in poly.items() if abs(value) > 1e-10}


def _add(left: Polynomial, right: Polynomial, scale: float = 1.0) -> Polynomial:
    result = dict(left)
    for power, value in right.items():
        result[power] = result.get(power, 0.0) + scale * value
    return _clean(result)


def _multiply(left: Polynomial, right: Polynomial) -> Polynomial:
    result: Polynomial = {}
    for (a1, a2), left_value in left.items():
        for (b1, b2), right_value in right.items():
            power = (a1 + b1, a2 + b2)
            if power[0] > 2 or power[1] > 2 or sum(power) > 2:
                raise ValueError("Equation exceeds the supported degree")
            result[power] = result.get(power, 0.0) + left_value * right_value
    return _clean(result)


def _power(poly: Polynomial, exponent: int) -> Polynomial:
    if exponent < 0 or exponent > 2:
        raise ValueError("Only powers 0, 1, and 2 are allowed")
    result: Polynomial = {(0, 0): 1.0}
    for _ in range(exponent):
        result = _multiply(result, poly)
    return result


def parse_polynomial(expression: str) -> Polynomial:
    """Parse the allowlisted equation grammar into polynomial coefficients."""
    if not isinstance(expression, str) or len(expression) > 256:
        raise ValueError("Equation must be a short string")
    tree = ast.parse(expression.strip(), mode="eval")

    def visit(node: ast.AST) -> Polynomial:
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                raise ValueError("Only numeric constants are allowed")
            value = float(node.value)
            if not math.isfinite(value) or abs(value) > 100:
                raise ValueError("Constant is out of bounds")
            return {(0, 0): value}
        if isinstance(node, ast.Name) and node.id in VARIABLES:
            return {(1, 0) if node.id == "x1" else (0, 1): 1.0}
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = visit(node.operand)
            return value if isinstance(node.op, ast.UAdd) else {key: -v for key, v in value.items()}
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
            return _add(visit(node.left), visit(node.right), -1.0 if isinstance(node.op, ast.Sub) else 1.0)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
            return _multiply(visit(node.left), visit(node.right))
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
            if not isinstance(node.right, ast.Constant) or isinstance(node.right.value, bool):
                raise ValueError("Power must be a literal integer")
            exponent = node.right.value
            if not isinstance(exponent, int):
                raise ValueError("Power must be a literal integer")
            return _power(visit(node.left), exponent)
        raise ValueError("Equation contains an unsupported operation")

    return _clean(visit(tree.body))


def evaluate_polynomial(poly: Mapping[Exponent, float], x1: float, x2: float) -> float:
    return sum(value * (x1**p1) * (x2**p2) for (p1, p2), value in poly.items())


def encode_polynomial(poly: Mapping[Exponent, float]) -> dict[str, float]:
    return {f"{p1},{p2}": float(value) for (p1, p2), value in sorted(poly.items())}


def decode_polynomial(data: Mapping[str, Any]) -> Polynomial:
    result: Polynomial = {}
    for key, value in data.items():
        p1, p2 = (int(part) for part in key.split(","))
        result[(p1, p2)] = float(value)
    return _clean(result)


def canonical_equation(poly: Mapping[Exponent, float]) -> str:
    labels = {
        (0, 0): "1",
        (1, 0): "x1",
        (0, 1): "x2",
        (2, 0): "x1**2",
        (1, 1): "x1*x2",
        (0, 2): "x2**2",
    }
    parts: list[str] = []
    for power in ((0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2)):
        coefficient = poly.get(power, 0.0)
        if abs(coefficient) <= 1e-10:
            continue
        label = labels[power]
        magnitude = abs(coefficient)
        if power == (0, 0):
            body = f"{magnitude:g}"
        elif math.isclose(magnitude, 1.0):
            body = label
        else:
            body = f"{magnitude:g}*{label}"
        if not parts:
            parts.append(body if coefficient > 0 else f"-{body}")
        else:
            parts.append((" + " if coefficient > 0 else " - ") + body)
    return "".join(parts) or "0"


def generate_universe(level: int, seed: int, budget: int, domain: tuple[int, int]) -> dict[str, Any]:
    if level not in BASES:
        raise ValueError("The implemented curriculum currently supports levels 1, 2, and 3")
    rng = random.Random(seed)
    polynomial: Polynomial = {}
    for power in BASES[level]:
        if power == (0, 0):
            polynomial[power] = float(rng.randint(-3, 3))
        else:
            polynomial[power] = float(rng.choice(COEFFICIENTS))
    return {
        "level": level,
        "seed": seed,
        "family": FAMILY_NAMES[level],
        "coefficients": encode_polynomial(polynomial),
        "equation": canonical_equation(polynomial),
        "budget": budget,
        "domain": list(domain),
    }


def _prompt(level: int, budget: int, domain: tuple[int, int]) -> str:
    family_hint = {
        1: "an affine law using 1, x1, and x2",
        2: "an additive polynomial using 1, x1, x2, x1**2, and x2**2",
        3: "a degree-two polynomial that may also contain x1*x2",
    }[level]
    return f"""You are The Scientist. A hidden deterministic universe maps integer inputs x1 and x2 to y.

This is Level {level}: the law is {family_hint}. Coefficients are small integers. Inputs must be integers in [{domain[0]}, {domain[1]}]. You have at most {budget} laboratory experiments.

Use the experiment tool to choose informative measurements. When ready, stop calling tools and reply with only one JSON object:
{{"family":"{FAMILY_NAMES[level]}","equation":"your expression","confidence":0.0}}

Allowed equation syntax: x1, x2, numbers, +, -, *, and powers **0 through **2. Do not include markdown fences."""


def build_dataset(
    level: int,
    count: int,
    seed: int,
    split: str,
    budget: int,
    domain: tuple[int, int],
) -> Dataset:
    split_offset = {"train": 0, "eval": 1_000_000, "test": 2_000_000}.get(split)
    if split_offset is None:
        raise ValueError("split must be train, eval, or test")
    rows = []
    for index in range(count):
        universe_seed = seed + split_offset + index
        universe = generate_universe(level, universe_seed, budget, domain)
        rows.append(
            {
                "question": _prompt(level, budget, domain),
                "answer": json.dumps(universe, sort_keys=True),
                "info": {"split": split, "universe_id": f"{split}-{level}-{universe_seed}"},
            }
        )
    return Dataset.from_list(rows)


def _last_assistant_text(completion: Any) -> str:
    if not isinstance(completion, list):
        return ""
    for message in reversed(completion):
        role = message.get("role") if isinstance(message, dict) else getattr(message, "role", None)
        if role != "assistant":
            continue
        content = message.get("content") if isinstance(message, dict) else getattr(message, "content", None)
        if isinstance(content, str):
            return content.strip()
    return ""


def parse_final_answer(completion: Any) -> tuple[dict[str, Any], Polynomial]:
    text = _last_assistant_text(completion)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("Final answer is not a JSON object")
    payload = json.loads(text[start : end + 1])
    if not isinstance(payload, dict) or not isinstance(payload.get("equation"), str):
        raise ValueError("Final answer must contain an equation string")
    confidence = payload.get("confidence")
    if confidence is not None and (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not 0 <= float(confidence) <= 1
    ):
        raise ValueError("confidence must lie between 0 and 1")
    return payload, parse_polynomial(payload["equation"])


def _truth(state: vf.State | Mapping[str, Any]) -> dict[str, Any]:
    answer = state.get("answer", "")
    if isinstance(answer, str):
        return json.loads(answer)
    if isinstance(answer, dict):
        return answer
    raise ValueError("Missing universe truth")


async def hidden_probe_score(completion: vf.Messages, state: vf.State) -> float:
    try:
        _, prediction = parse_final_answer(completion)
        truth = _truth(state)
        target = decode_polynomial(truth["coefficients"])
        low, high = (int(value) for value in truth["domain"])
        points = [(x1, x2) for x1 in range(low, high + 1) for x2 in range(low, high + 1)]
        rng = random.Random(int(truth["seed"]) + 90_001)
        probes = rng.sample(points, min(16, len(points)))
        errors = [
            abs(evaluate_polynomial(prediction, x1, x2) - evaluate_polynomial(target, x1, x2))
            for x1, x2 in probes
        ]
        scale = 1.0 + sum(abs(evaluate_polynomial(target, x1, x2)) for x1, x2 in probes) / len(probes)
        return max(0.0, 1.0 - (sum(errors) / len(errors)) / scale)
    except (ValueError, TypeError, KeyError, json.JSONDecodeError, SyntaxError, OverflowError):
        return 0.0


async def structure_score(completion: vf.Messages, state: vf.State) -> float:
    try:
        payload, prediction = parse_final_answer(completion)
        truth = _truth(state)
        target = decode_polynomial(truth["coefficients"])
        powers = set(target) | set(prediction)
        coefficient_score = sum(
            math.isclose(target.get(power, 0.0), prediction.get(power, 0.0), abs_tol=1e-6)
            for power in powers
        ) / max(1, len(powers))
        family_score = float(payload.get("family") == truth["family"])
        return 0.8 * coefficient_score + 0.2 * family_score
    except (ValueError, TypeError, KeyError, json.JSONDecodeError, SyntaxError, OverflowError):
        return 0.0


async def valid_final_answer(completion: vf.Messages) -> float:
    try:
        parse_final_answer(completion)
        return 1.0
    except (ValueError, TypeError, KeyError, json.JSONDecodeError, SyntaxError, OverflowError):
        return 0.0


async def experiments_used(state: vf.State) -> float:
    return float(len(state.get("lab_state", {}).get("observations", [])))


async def duplicate_query_rate(state: vf.State) -> float:
    observations = state.get("lab_state", {}).get("observations", [])
    if not observations:
        return 0.0
    return sum(bool(item.get("duplicate")) for item in observations) / len(observations)


class ScientistEnv(vf.StatefulToolEnv):
    """A per-rollout hidden universe with a strictly budgeted experiment tool."""

    def __init__(self, *, budget: int, domain: tuple[int, int], **kwargs: Any):
        self.budget = budget
        self.domain = domain
        rubric = vf.Rubric(funcs=[hidden_probe_score, structure_score], weights=[0.8, 0.2])
        rubric.add_metric(valid_final_answer)
        rubric.add_metric(experiments_used)
        rubric.add_metric(duplicate_query_rate)
        super().__init__(
            tools=[],
            rubric=rubric,
            max_turns=budget + 1,
            error_formatter=lambda _: "The laboratory rejected that tool call.",
            **kwargs,
        )
        self.add_tool(self.experiment, args_to_skip=["lab_state"])

    async def setup_state(self, state: vf.State) -> vf.State:
        state = await super().setup_state(state) or state
        truth = _truth(state)
        state["lab_state"] = {
            "polynomial": decode_polynomial(truth["coefficients"]),
            "remaining": int(truth["budget"]),
            "domain": tuple(int(value) for value in truth["domain"]),
            "observations": [],
        }
        return state

    def update_tool_args(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        messages: vf.Messages,
        state: vf.State,
        **kwargs: Any,
    ) -> dict[str, Any]:
        updated = dict(tool_args)
        if tool_name == "experiment":
            updated["lab_state"] = state["lab_state"]
        return updated

    def experiment(self, x1: int, x2: int, lab_state: dict[str, Any]) -> str:
        """Run one experiment in the hidden universe.

        Args:
            x1: Integer first input within the declared domain.
            x2: Integer second input within the declared domain.
            lab_state: Private rollout state injected by the environment.
        """
        remaining = int(lab_state["remaining"])
        if remaining <= 0:
            return json.dumps({"error": "experiment budget exhausted", "experiments_remaining": 0})
        lab_state["remaining"] = remaining - 1

        values: list[int] = []
        for value in (x1, x2):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not float(value).is_integer():
                return json.dumps(
                    {"error": "inputs must be integers", "experiments_remaining": lab_state["remaining"]}
                )
            values.append(int(value))
        low, high = lab_state["domain"]
        if any(value < low or value > high for value in values):
            return json.dumps(
                {"error": f"inputs must lie in [{low}, {high}]", "experiments_remaining": lab_state["remaining"]}
            )

        query = tuple(values)
        duplicate = any(tuple(item["x"]) == query for item in lab_state["observations"] if "x" in item)
        y = evaluate_polynomial(lab_state["polynomial"], *query)
        observation = {"x": list(query), "y": y, "duplicate": duplicate}
        lab_state["observations"].append(observation)
        return json.dumps(
            {"y": y, "duplicate": duplicate, "experiments_remaining": lab_state["remaining"]},
            sort_keys=True,
        )


def load_environment(
    level: int = 1,
    num_train: int = 512,
    num_eval: int = 128,
    seed: int = 17,
    budget: int = 6,
    domain_min: int = -3,
    domain_max: int = 3,
) -> vf.Environment:
    """Load The Scientist environment.

    Args:
        level: Curriculum level, currently 1 through 3.
        num_train: Number of procedurally generated training universes.
        num_eval: Number of disjoint evaluation universes.
        seed: Base seed for deterministic generation.
        budget: Maximum experiments per episode.
        domain_min: Inclusive lower bound for integer experiment inputs.
        domain_max: Inclusive upper bound for integer experiment inputs.
    """
    if budget < 1:
        raise ValueError("budget must be positive")
    if domain_min >= domain_max:
        raise ValueError("domain_min must be smaller than domain_max")
    domain = (domain_min, domain_max)
    return ScientistEnv(
        budget=budget,
        domain=domain,
        dataset=build_dataset(level, num_train, seed, "train", budget, domain),
        eval_dataset=build_dataset(level, num_eval, seed, "eval", budget, domain),
    )
