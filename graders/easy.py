"""Easy grader: single API doc -> at least 5 scenarios. Score strictly in (0, 1)."""


def grade(result: dict) -> float:
    # Handle multiple data formats the platform may pass:
    # - scenario_list: list (internal pipeline format)
    # - scenarios: list (observation format)
    # - scenarios_generated: int (state/observation count field)
    scenario_list = result.get("scenario_list") or result.get("scenarios") or []
    if isinstance(scenario_list, list):
        count = len(scenario_list)
    else:
        count = 0

    # Fall back to integer count field if list is empty
    if count == 0:
        count = int(result.get("scenarios_generated", 0))

    if count >= 5:
        raw = 0.99
    elif count == 0:
        raw = 0.01
    else:
        raw = round(max(0.01, min(count / 5, 0.99)), 2)

    return raw
