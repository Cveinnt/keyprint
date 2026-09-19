import importlib.util
import math
from pathlib import Path
import sys

import pytest


@pytest.fixture
def study(monkeypatch):
    tools = Path(__file__).parents[1] / "tools"
    for name in ("compare_score_baselines", "validate_null_corpus"):
        spec = importlib.util.spec_from_file_location(name, tools / (name + ".py"))
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
    return module


def test_transitive_duplicate_groups_cannot_cross_splits(study):
    def row(instruction, context, text):
        return dict(instruction=instruction, context=context, response=(text + " ") * 100, category="qa")
    rows = [row("Alpha", "ctx one", "one"), row(" alpha ", "ctx two", "two"),
            row("Beta", "ctx two", "three"), row("Gamma", "", "four"), row("Delta", "", "five")]
    selected, counts = study.select_rows(rows, 3)
    assert counts["independent_exact_groups"] == 3
    assert len({r["source_index"] for r in selected} & {0, 1, 2}) == 1
    assert study.select_rows(rows, 3) == (selected, counts)
    with pytest.raises(ValueError, match="Insufficient"):
        study.select_rows(rows, 4)


def test_small_calibration_is_rejected_and_ties_are_not_exceedances(study):
    with pytest.raises(ValueError, match="Insufficient"):
        study.threshold(list(range(12)), .01)
    assert study.threshold(list(range(500)), .01) == 495
    row = {"category": "qa", "maximum": {"uniform": 3.0}}
    summary = study.summarize([row, {"category": "qa", "error": "ValueError"}], {"uniform": 3.0}, 2)["uniform"]
    assert summary["above"] == 0 and summary["unavailable"] == 1
    assert summary["iid_only_upper_97_5_percent"] is None


def test_exact_binomial_upper_and_missing_attempts(study):
    assert study.iid_binomial_upper(0, 500) == pytest.approx(1 - .025 ** (1 / 500))
    assert study.iid_binomial_upper(5, 5) == 1
    p = study.iid_binomial_upper(1, 500)
    assert (1 - p) ** 500 + 500 * p * (1 - p) ** 499 == pytest.approx(.025)
    assert study.summarize([], {"uniform": 2.0}, 500)["uniform"]["unavailable"] == 500
    with pytest.raises(ValueError):
        study.threshold([math.nan] * 500, .01)
