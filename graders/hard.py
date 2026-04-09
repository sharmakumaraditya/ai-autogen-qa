"""Hard grader: full pipeline with test cases and quality score. Score strictly in (0, 1)."""


def grade(result: dict) -> float:
    scenarios = result.get("scenario_list", [])
    test_cases = result.get("test_details_list", [])
    quality = result.get("stage2_quality_score", 0.0)
    if quality is None:
        quality = 0.0
    finished_s1 = bool(result.get("is_finished_stage1", False))
    finished_s2 = bool(result.get("is_finished_stage2", False))

    sc = len(scenarios) if isinstance(scenarios, list) else 0
    tc = len(test_cases) if isinstance(test_cases, list) else 0

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

    raw = min(round(score, 2), 1.0)
    # Must be strictly between 0 and 1 (exclusive)
    return max(0.01, min(raw, 0.99))
