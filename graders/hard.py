"""Hard grader: full pipeline with test cases and quality score. Score strictly in (0, 1)."""


def grade(result: dict) -> float:
    # Handle multiple data formats the platform may pass
    scenario_list = result.get("scenario_list") or result.get("scenarios") or []
    sc = len(scenario_list) if isinstance(scenario_list, list) else 0
    if sc == 0:
        sc = int(result.get("scenarios_generated", 0))

    test_list = result.get("test_details_list") or result.get("test_cases") or []
    tc = len(test_list) if isinstance(test_list, list) else 0
    if tc == 0:
        tc = int(result.get("test_cases_generated", 0))

    quality = result.get("stage2_quality_score") or result.get("quality_score") or 0.0
    if quality is None:
        quality = 0.0
    quality = float(quality)

    finished_s1 = bool(result.get("is_finished_stage1", False))
    finished_s2 = bool(result.get("is_finished_stage2", False))

    score = 0.0

    # Scenario coverage (25%)
    if sc >= 10:
        score += 0.25
    elif sc >= 5:
        score += round(0.25 * sc / 10, 2)

    # Stage 1 reflection pass (15%)
    if finished_s1:
        score += 0.15

    # Test case generation (25%)
    if tc >= 10:
        score += 0.25
    elif tc > 0:
        score += round(0.25 * tc / 10, 2)

    # Stage 2 reflection pass (10%)
    if finished_s2:
        score += 0.10

    # Quality score (25%)
    score += round(0.25 * min(float(quality), 1.0), 2)

    return max(0.01, min(round(score, 2), 0.99))
