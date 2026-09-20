import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location("serving_benchmark",Path(__file__).parents[1]/"tools/benchmark_serving.py")
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


def fixture(ratio=1.02):
    return [{"case":case,"repeat":repeat,"condition":condition,
             "seconds":2*ratio if condition=="marked" else 1.,
             "completion_tokens":200 if condition=="marked" else 100}
            for case in ("a","b") for repeat in range(benchmark.REPEATS)
            for condition in ("ordinary","marked")]


def test_actual_token_cost_is_distinct_from_whole_request_latency():
    result = benchmark.summarize(fixture(),["a","b"])
    assert result["pairs"]==8
    assert result["seconds_per_committed_token"]["geometric_mean_ratio"]==pytest.approx(1.02)
    assert result["request_latency"]["geometric_mean_ratio"]==pytest.approx(2.04)
    assert result["incremental_5pct_timing_screen_pass"] is True
    assert result["production_overhead_accepted"] is False
    assert result["native_server_overhead_measured"] is False
    assert benchmark.summarize(fixture(1.06),["a","b"])["incremental_5pct_timing_screen_pass"] is False


def test_paired_uncertainty_is_reproducible_and_not_just_point_estimate():
    rows = fixture(.8)
    rows[1]["seconds"]=10.
    first = benchmark.summarize(rows,["a","b"])
    assert first==benchmark.summarize(rows,["a","b"])
    result = first["seconds_per_committed_token"]
    assert result["one_sided_95_upper_ratio"]>result["geometric_mean_ratio"]
    assert first["incremental_5pct_timing_screen_pass"] is False


@pytest.mark.parametrize("change",["missing","duplicate","failure","zero_tokens","zero_time","nan","boolean_time"])
def test_bad_or_omitted_measurements_cannot_pass(change):
    rows = fixture()
    if change=="missing": rows.pop()
    if change=="duplicate": rows[-1]=dict(rows[0])
    if change=="failure": rows[-1]["error"]={"type":"Failed"}
    if change=="zero_tokens": rows[-1]["completion_tokens"]=0
    if change=="zero_time": rows[-1]["seconds"]=0.
    if change=="nan": rows[-1]["seconds"]=float("nan")
    if change=="boolean_time": rows[-1]["seconds"]=True
    with pytest.raises(ValueError):
        benchmark.summarize(rows,["a","b"])


@pytest.mark.parametrize("case_ids", [[], ["a", "a"]])
def test_invalid_workload_cannot_receive_a_timing_summary(case_ids):
    with pytest.raises(ValueError, match="Nonempty distinct"):
        benchmark.summarize([], case_ids)


def test_storage_preflight_uses_destination_without_creating_output(tmp_path, monkeypatch):
    output = tmp_path / "missing" / "run"
    checked = []
    def usage(path):
        checked.append(path)
        return SimpleNamespace(free=benchmark.MIN_FREE_BYTES - 1)
    monkeypatch.setattr(benchmark.shutil, "disk_usage", usage)
    with pytest.raises(ValueError, match="at least 2 GiB"):
        benchmark.require_storage(output)
    assert checked == [tmp_path]
    assert not output.parent.exists()
    monkeypatch.setattr(benchmark.shutil, "disk_usage", lambda _: SimpleNamespace(free=benchmark.MIN_FREE_BYTES))
    assert benchmark.require_storage(output) == benchmark.MIN_FREE_BYTES
