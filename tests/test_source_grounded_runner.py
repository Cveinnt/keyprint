import json
from pathlib import Path
import sys
import hashlib

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "tools"))
from validate_source_grounded import schedule, load_inputs


def inputs(tmp_path):
    cases = [{"id": str(i)} for i in range(16)]
    rubrics = [{"id": str(i), "essential_facts": ["pending"], "format": "direct", "qualifiers": "not complete"} for i in range(16)]
    (tmp_path / "private").mkdir(); (tmp_path / "public").mkdir()
    commitment = {}
    for name, rows in [("cases.json", cases), ("rubrics.json", rubrics)]:
        data = json.dumps(rows).encode()
        (tmp_path / "private" / name).write_bytes(data)
        commitment[name] = hashlib.sha256(data).hexdigest()
    (tmp_path / "public/rubrics-commitment.json").write_text(json.dumps(commitment))
    (tmp_path / "public/selection.json").write_text(json.dumps({"private_cases_sha256": commitment["cases.json"]}))


def test_all_four_keys_and_conditions_are_scheduled_without_selection():
    rows = schedule([{"id": str(i)} for i in range(16)])
    assert len(rows) == 128
    assert len({(r["case"], r["key_slot"], r["condition"]) for r in rows}) == 128
    assert rows[0]["condition"] == "ordinary" and rows[2]["condition"] == "marked"
    assert rows[32]["condition"] == "marked"


@pytest.mark.parametrize("change", ["source", "rubric", "selection"])
def test_changed_preregistered_input_is_rejected(tmp_path, change):
    inputs(tmp_path)
    assert len(load_inputs(tmp_path)[0]) == 16
    if change == "source": (tmp_path / "private/cases.json").write_text("[]")
    elif change == "rubric": (tmp_path / "private/rubrics.json").write_text("[]")
    else: (tmp_path / "public/selection.json").write_text('{"private_cases_sha256":"wrong"}')
    with pytest.raises(ValueError):
        load_inputs(tmp_path)


def test_missing_or_duplicate_cases_cannot_shrink_the_study():
    with pytest.raises(ValueError): schedule([{"id": str(i)} for i in range(15)])
    with pytest.raises(ValueError): schedule([{"id": "same"}] * 16)
