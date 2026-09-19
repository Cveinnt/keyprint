import importlib.util
from pathlib import Path
import sys

import pytest


@pytest.fixture
def study(monkeypatch):
    tools = Path(__file__).parents[1] / "tools"
    for name in ("compare_score_baselines", "validate_null_corpus", "validate_corpus_power"):
        spec = importlib.util.spec_from_file_location(name, tools / (name + ".py"))
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
    return module


def test_selection_uses_only_heldout_source_tasks_and_checks_binding(study):
    source, records = [], []
    for category in study.CATEGORIES:
        for i in range(3):
            row = dict(category=category, instruction=f"task {category} {i}", context="source paragraph", response="reference answer")
            records.append(dict(source_index=len(source), split="calibration" if i == 0 else "heldout",
                                text_sha256=study.sha(row['response'].encode())))
            source.append(row)
    selected = study.select_tasks(source, {"records": records})
    assert len(selected) == 12
    assert all(r["source_index"] % 3 != 0 for r in selected)
    assert all(r["prompt"].endswith("Reference text:\nsource paragraph") for r in selected)
    source[1]["response"] = "tampered reference"
    with pytest.raises(ValueError, match="hash differs"):
        study.select_tasks(source, {"records": records})


def test_summary_separates_matching_wrong_key_and_ordinary_max(study):
    rows = [{"condition": "marked", "scores": {"matching": {"uniform": 3}, "other": {"uniform": 1}}},
            {"condition": "marked", "error": "KeyprintError"},
            {"condition": "ordinary", "scores": {"maximum": {"uniform": 2}}}]
    groups = study.summarize(rows, {"uniform": 2})["uniform"]["groups"]
    assert groups["marked_matching"] == dict(attempts=2, available=1, unavailable=1, above=1)
    assert groups["marked_other"]["above"] == 0
    assert groups["ordinary_maximum"]["above"] == 0
