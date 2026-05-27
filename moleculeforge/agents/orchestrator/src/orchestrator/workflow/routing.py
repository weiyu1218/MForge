"""Conditional routing functions for the orchestrator workflow."""


def route_after_validation(state: dict) -> str:
    if state.get("validation_passed", False):
        return "fto_check"
    return "refine"


def route_after_critic(state: dict) -> str:
    if state.get("critic_passed", False):
        return "orchestrate"
    return "refine"


def orchestrator_decision(state: dict) -> str:
    cycle = state.get("cycle_count", 0)
    max_cycles = state.get("max_cycles", 20)
    if cycle >= max_cycles:
        return "accept"
    if state.get("all_passed", False):
        return "accept"
    return "refine"
