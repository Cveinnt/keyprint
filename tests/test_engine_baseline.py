from copy import deepcopy
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "tools"))
try:
    from benchmark_engine_baseline import ARMS, analyze, collect_engine, order
finally:
    sys.path.pop(0)


def response(token, count, finish=None, text="x"):
    return SimpleNamespace(token=token, generation_tokens=count, finish_reason=finish, text=text)


def test_upstream_terminal_is_counted_once_including_eos():
    result = collect_engine([response(1, 1), response(2, 2, "stop", "")], cap=8, eos_ids={2})
    assert result == {"token_ids": [1, 2], "text": "x", "completion_tokens": 2, "completion": "eos"}
    capped = collect_engine([response(1, 1, "length")], cap=1, eos_ids={2})
    assert capped["completion_tokens"] == 1 and capped["completion"] == "length"


@pytest.mark.parametrize("items", [[], [response(1, 1)], [response(1, 2, "length")],
    [response(1, 1, "stop")], [response(2, 1, "length")],
    [response(1, 1, "length"), response(1, 2, "length")]])
def test_incomplete_or_inconsistent_engine_receipt_cannot_be_timed_as_success(items):
    with pytest.raises(ValueError):
        collect_engine(items, cap=1, eos_ids={2})


def rows():
    return [{"case": case, "repeat": repeat, "arm": arm, "seconds": seconds,
             "completion_tokens": tokens}
            for case in ["a", "b"] for repeat in range(4)
            for arm, seconds, tokens in [("engine", 1, 10), ("sdk_ordinary", 2, 10),
                                          ("sdk_marked", 4.2, 20)]]


def test_ratios_use_actual_tokens_and_do_not_conflate_marking_with_sdk_overhead():
    result = analyze(rows(), ["a", "b"])["comparisons"]
    assert result["sdk_marked_over_engine"]["seconds_per_committed_token"]["geometric_mean_ratio"] == pytest.approx(2.1)
    assert result["sdk_marked_over_sdk_ordinary"]["seconds_per_committed_token"]["geometric_mean_ratio"] == pytest.approx(1.05)
    assert result["sdk_marked_over_engine"]["local_5pct_screen_pass"] is False


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "error", "zero_tokens"])
def test_failed_or_replaced_arm_cannot_enter_summary(mutation):
    values = deepcopy(rows())
    if mutation == "missing": values.pop()
    elif mutation == "duplicate": values[-1] = values[0]
    elif mutation == "error": values[0]["error"] = {"type": "RuntimeError"}
    else: values[0]["completion_tokens"] = 0
    with pytest.raises(ValueError): analyze(values, ["a", "b"])


def test_rotated_order_covers_all_three_positions():
    for arm in ARMS:
        assert sorted(order(0, i).index(arm) for i in range(3)) == [0, 1, 2]
