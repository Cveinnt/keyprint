import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("baseline", Path(__file__).parents[1] / "tools/validate_model_baseline.py")
baseline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(baseline)


def test_schedule_retains_both_attempts_per_case_and_unique_seeds():
    rows = baseline.schedule([{"id": name} for name in "abcd"])
    assert len(rows) == len({r["seed"] for r in rows}) == 8
    assert [(r["case"], r["repetition"]) for r in rows] == [(c, i) for i in range(2) for c in "abcd"]


def events(reason="stop", token=2):
    return [{"text": " A", "token": 1, "generation_tokens": 1, "finish_reason": None},
            {"text": "", "token": token, "generation_tokens": 2, "finish_reason": reason}]


def test_native_text_and_eos_are_retained_without_whitespace_repair():
    r = baseline.reconcile(events(), lambda ids: " A" if ids == [1] else "bad", {2}, 2)
    assert r["text"] == " A" and r["sampled_token_ids"] == [1, 2]
    assert r["visible_token_ids"] == [1] and r["completion"] == "eos"
    with pytest.raises(ValueError, match="Stream differs"):
        baseline.reconcile(events(), lambda ids: "A", {2}, 2)


def test_cap_is_not_reclassified_as_successful_stop():
    r = baseline.reconcile(events("length", 3), lambda ids: " A", {2}, 2)
    assert r["completion"] == "length" and r["visible_token_ids"] == [1, 3]


@pytest.mark.parametrize("change", ["wrong_eos", "bad_count", "early_stop", "short_cap"])
def test_incomplete_or_inconsistent_receipts_fail(change):
    e = events()
    if change == "wrong_eos": e[-1]["token"] = 3
    if change == "bad_count": e[-1]["generation_tokens"] = 4
    if change == "early_stop": e[0]["finish_reason"] = "stop"
    if change == "short_cap": e[-1].update(token=3, finish_reason="length")
    with pytest.raises(ValueError):
        baseline.reconcile(e, lambda ids: " A", {2}, 8)
