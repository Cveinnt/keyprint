from pathlib import Path
import sys

import pytest

TOOLS = Path(__file__).parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
try:
    import benchmark_fast_serving as benchmark
finally:
    sys.path.remove(str(TOOLS))


def fixture():
    return [{"execution": execution, "condition": condition, "case": case, "repeat": repeat,
             "completion_tokens": 200 if condition == "marked" else 100,
             "seconds": (2.08 if condition == "marked" else 1.) * (.5 if execution == "fast" else 1.)}
            for execution, condition in benchmark.CELLS for case in ("a", "b") for repeat in range(4)]


def test_speedup_is_separate_from_incremental_watermark_cost():
    report = benchmark.analyze(fixture(), ["a", "b"])
    assert report["primary_fast_5pct_screen_pass"] is True
    for execution in ("reference", "fast"):
        assert report["within_execution"][execution]["seconds_per_committed_token"]["geometric_mean_ratio"] == pytest.approx(1.04)
    for condition in ("ordinary", "marked"):
        assert report["fast_over_reference"][condition]["geometric_mean_seconds_per_token_ratio"] == pytest.approx(.5)
    rows = fixture()
    for row in rows:
        if row["execution"] == "fast" and row["condition"] == "marked": row["seconds"] *= 1.2
    failed = benchmark.analyze(rows, ["a", "b"])
    assert failed["primary_fast_5pct_screen_pass"] is False
    assert failed["fast_over_reference"]["marked"]["geometric_mean_seconds_per_token_ratio"] < 1
    assert failed["production_overhead_accepted"] is False


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "error", "zero", "unknown_execution"])
def test_bad_or_missing_cells_are_not_excluded(mutation):
    rows = fixture()
    if mutation == "missing": rows.pop()
    elif mutation == "duplicate": rows[-1] = dict(rows[0])
    elif mutation == "error": rows[-1]["error"] = {"type": "Failure"}
    elif mutation == "zero": rows[-1]["completion_tokens"] = 0
    else: rows[-1]["execution"] = "other"
    with pytest.raises(ValueError): benchmark.analyze(rows, ["a", "b"])


def test_each_execution_condition_visits_every_order_position():
    for case_index in range(6):
        for position in range(4):
            assert {benchmark.cell_order(repeat, case_index)[position] for repeat in range(4)} == set(benchmark.CELLS)
