from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "tools"))
from summarize_source_grounded import diagnostics, summarize
from validate_source_grounded import schedule


def inputs():
    cases = [{"id": str(i), "essential_facts": ["first", "second"]} for i in range(16)]
    plan = {"schedule": schedule(cases)}
    rows = [dict(r, review_id=str(i), completion="eos") for i, r in enumerate(plan["schedule"])]
    ratings = [{"review_id": r["review_id"], "fact_checks": [True, True], "no_unsupported_claims": True,
                "language_pass": True, "format_pass": True, "uncertain_fields": [], "reason": "Both source facts present; no added claims."} for r in rows]
    return plan, rows, ratings, cases


def test_coverage_claims_and_format_remain_separate_and_paired():
    plan, rows, ratings, cases = inputs()
    ratings[0]["fact_checks"][0] = False  # ordinary omission
    ratings[0]["uncertain_fields"] = ["fact:0"]
    ratings[1]["no_unsupported_claims"] = False  # marked invention
    ratings[2]["format_pass"] = False  # next marked format
    result = summarize(plan, rows, ratings, cases)
    assert result["groups"]["ordinary"]["supported_claims"] == 64
    assert result["groups"]["ordinary"]["essential_facts"] == 63
    assert result["groups"]["marked"]["essential_facts"] == 64
    assert result["groups"]["marked"]["supported_claims"] == 63
    assert result["groups"]["marked"]["full_task"] == 62
    assert result["paired"]["full_task"] == {"both_pass": 62, "ordinary_only": 1, "marked_only": 0, "neither_pass": 1}
    generous = summarize(plan, rows, ratings, cases, accept_uncertain=True)
    assert generous["groups"]["ordinary"]["content"] == 64
    assert generous["groups"]["marked"]["content"] == 63  # unflagged invention remains
    assert ratings[0]["fact_checks"][0] is False


@pytest.mark.parametrize("change", ["missing", "duplicate", "unknown", "unanswered", "invalid_flag", "missing_fact", "no_reason", "failure_pass"])
def test_incomplete_or_invalid_ratings_rejected(change):
    plan, rows, ratings, cases = inputs()
    if change == "missing": ratings.pop()
    elif change == "duplicate": ratings[0] = ratings[1]
    elif change == "unknown": ratings[0]["review_id"] = "unknown"
    elif change == "unanswered": ratings[0]["language_pass"] = None
    elif change == "invalid_flag": ratings[0]["uncertain_fields"] = ["fact:9"]
    elif change == "missing_fact": ratings[0]["fact_checks"].pop()
    elif change == "no_reason": ratings[0]["reason"] = ""
    else: rows[0]["error_type"] = "RuntimeError"
    with pytest.raises(ValueError): summarize(plan, rows, ratings, cases)


def test_missing_detector_result_is_unavailable_not_negative_detection():
    _, rows, _, _ = inputs()
    rows[0]["raw_counts"] = [{"events": 2, "ones": 4, "trials": 6}, {"unavailable": "nonliteral"}]
    result = diagnostics(rows)
    assert result["ordinary"]["matching"] == {"available": 1, "unavailable": 63, "ones": 4, "trials": 6, "events": 2}
    assert result["ordinary"]["next_key"]["unavailable"] == 64
    assert "hits" not in str(result)


def test_token_caps_stay_in_denominator_and_cannot_pass_full_task():
    plan, rows, ratings, cases = inputs()
    rows[0]["completion"] = "length"
    result = summarize(plan, rows, ratings, cases)
    assert result["groups"]["ordinary"]["attempts"] == 64
    assert result["groups"]["ordinary"]["content"] == 64
    assert result["groups"]["ordinary"]["full_task"] == 63
