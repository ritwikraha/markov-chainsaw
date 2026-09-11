from zero_ad_bench.adapters import ZeroADEnvironment, normalize_state


class Unit:
    def __init__(self, entity_id, owner, template, position=None):
        self.data = {"id": entity_id}
        if position is not None:
            self.data["position"] = position
        self.owner = owner
        self.template = template
        self._position = position

    def id(self):
        return self.data["id"]

    def type(self):
        return self.template

    def position(self):
        return self._position


class State:
    def __init__(self):
        self.data = {
            "timeElapsed": 1000,
            "players": [
                {},
                {"state": "active", "phase": "village", "resourceCounts": {"wood": 300}, "resourceGatherers": {}, "popCount": 1, "popLimit": 20},
                {"state": "active", "phase": "village", "resourceCounts": {"wood": 300}, "resourceGatherers": {}, "popCount": 1, "popLimit": 20},
            ],
            "benchmarkMap": {"explored_fraction": 0.1},
            "benchmarkEvents": {},
            "events": [],
        }
        self._units = [
            Unit(1, 1, "units/spart/support_civilian", [10, 20]),
            Unit(2, 2, "structures/spart/civil_centre", [80, 90]),
        ]

    def units(self, owner):
        return [unit for unit in self._units if unit.owner == owner]


def test_normalize_state_contains_scoring_telemetry():
    observation = normalize_state(State(), "episode", 5)
    assert observation["player"]["resources"]["wood"] == 300
    assert observation["entities"][0]["position"] == [10.0, 20.0]


def test_vision_observation_withholds_structured_state():
    environment = ZeroADEnvironment(
        game=None,
        actions_module=None,
        reset_payload_factory=lambda task, seed: {},
        track="vision",
        frame_provider=lambda step: f"frame-{step}.png",
    )
    environment.state = State()
    environment.episode_id = "episode"
    visible = environment._observation()
    evaluator = environment.evaluation_observation()
    assert visible["frame_path"] == "frame-0.png"
    assert "player" not in visible
    assert evaluator["player"]["resources"]["wood"] == 300
