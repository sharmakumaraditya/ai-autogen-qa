"""Easy grader: single API doc -> at least 5 scenarios. Score strictly in (0, 1)."""


def grade(result: dict) -> float:
    scenarios = result.get("scenario_list", [])
    count = len(scenarios) if isinstance(scenarios, list) else 0
    if count >= 5:
        raw = 1.0
    elif count <= 0:
        raw = 0.0
    else:
        raw = round(count / 5, 2)
    # Must be strictly between 0 and 1 (exclusive)
    return max(0.01, min(raw, 0.99))
