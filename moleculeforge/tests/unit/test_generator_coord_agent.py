from generator_coord.agent import GeneratorCoordAgent
from mf_core.routing.task_router import GENERATOR_NAMES


def test_available_generators_match_task_router_names():
    agent = GeneratorCoordAgent()

    assert agent.generators == list(GENERATOR_NAMES)
    assert set(agent.generators).issubset(GENERATOR_NAMES)


def test_select_all_returns_task_router_names():
    agent = GeneratorCoordAgent()

    selected = agent._select_generators("all", {})

    assert selected == list(GENERATOR_NAMES)
    assert set(selected).issubset(GENERATOR_NAMES)


def test_select_real_generator_strategy_returns_that_generator():
    agent = GeneratorCoordAgent()

    selected = agent._select_generators("hfm_3d", {})

    assert selected == ["hfm_3d"]
    assert set(selected).issubset(GENERATOR_NAMES)


def test_auto_selected_generators_are_task_router_names():
    agent = GeneratorCoordAgent()

    for objectives in (
        {"complexity": "high"},
        {"complexity": "low"},
        {"complexity": "medium"},
        {},
    ):
        selected = agent._select_generators("auto", objectives)
        assert set(selected).issubset(GENERATOR_NAMES)


def test_unknown_strategy_uses_stable_real_generator_defaults():
    agent = GeneratorCoordAgent()

    selected = agent._select_generators("unknown", {})

    assert selected == ["hfm_3d", "fragfm"]
    assert set(selected).issubset(GENERATOR_NAMES)
