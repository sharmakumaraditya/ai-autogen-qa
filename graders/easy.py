"""Easy grader: single API doc -> at least 5 scenarios. Score 0.0-1.0."""


def grade(result: dict) -> float:
    scenarios = result.get("scenario_list", [])
    count = len(scenarios) if isinstance(scenarios, list) else 0
    if count >= 5:
        return 1.0
    if count <= 0:
        return 0.0
    return round(count / 5, 2)
