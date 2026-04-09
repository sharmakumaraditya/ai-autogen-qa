"""Medium grader: FRD + API doc -> 10+ scenarios with reflection pass. Score strictly in (0, 1)."""


def grade(result: dict) -> float:
    scenarios = result.get("scenario_list", [])
    reflected = bool(result.get("is_finished_stage1", False))

    count = len(scenarios) if isinstance(scenarios, list) else 0
    score = 0.0

    if count >= 10:
        score += 0.6
    elif count >= 5:
        score += round(0.6 * count / 10, 2)

    if reflected:
        score += 0.4

    raw = min(round(score, 2), 1.0)
    # Must be strictly between 0 and 1 (exclusive)
    return max(0.01, min(raw, 0.99))
