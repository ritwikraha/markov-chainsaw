from zero_ad_bench import AgentDecision, BenchmarkRunner
from zero_ad_bench.probes import ProbeEnvironment, default_probe_cases, summarize_probe_results


class CandidateAgent:
    name = "candidate-agent"

    def act(self, observation, legal_actions, context):
        target = next(
            action for action in observation["action_candidates"]
            if action == ProbeLookup.expected[context["task_id"]]
        )
        return AgentDecision(action=target, prompt_tokens=30, completion_tokens=10, latency_ms=5)


class ProbeLookup:
    expected = {case.id: case.expected_action for case in default_probe_cases()}


def test_probe_catalog_is_balanced():
    cases = default_probe_cases()
    assert len(cases) == 15
    assert {category: sum(case.category == category for case in cases) for category in {case.category for case in cases}} == {
        "economy": 3,
        "building-tech": 3,
        "scouting": 3,
        "combat": 3,
        "adaptive": 3,
    }


def test_probe_suite_records_perfect_agent(tmp_path):
    summaries = []
    runner = BenchmarkRunner(tmp_path)
    for case in default_probe_cases():
        summaries.append(runner.run(case.task(), CandidateAgent(), ProbeEnvironment(case), episode_id=case.id))
    result = summarize_probe_results(summaries)
    assert result["correct"] == 15
    assert result["accuracy"] == 1.0
    assert result["legal_action_rate"] == 1.0

