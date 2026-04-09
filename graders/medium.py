"""Medium grader: FRD + API doc -> 10+ scenarios with reflection pass. Score strictly in (0, 1)."""


def grade(result: dict) -> float:
    # Handle multiple data formats the platform may pass
    scenario_list = result.get("scenario_list") or result.get("scenarios") or []
    if isinstance(scenario_list, list):
        count = len(scenario_list)
    else:
        count = 0

    if count == 0:
        count = int(result.get("scenarios_generated", 0))

    reflected = bool(result.get("is_finished_stage1", False))

    score = 0.0
    if count >= 10:
        score += 0.6
    elif count >= 5:
        score += round(0.6 * count / 10, 2)

    if reflected:
        score += 0.4

    return max(0.01, min(round(score, 2), 0.99))
