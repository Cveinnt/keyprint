import json
from pathlib import Path
import re
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "tools"))
from build_source_review import build, replace_once


def rows():
    return [{"review_id": f"{i:012x}", "case": {"id": str(i // 8), "source": "Original source <script>bad()</script>",
        "instruction": "Summarize; no arbitrary length."}, "rubric": {"id": str(i // 8),
        "essential_facts": ["Only supplied claims"], "format": "Brief bullets", "qualifiers": "No inventions"},
        "text": "Output", "completion": "eos", "error_type": None} for i in range(128)]


def test_source_and_format_are_visible_without_script_injection_or_fake_word_limit():
    source = rows()
    source[0].update(error_type="KeyprintError", completion=None, text="")
    html = build(json.dumps(source))
    payload = json.loads(re.search(r'id="review-data">(.*?)</script>', html, re.S).group(1))
    assert len(payload["rows"]) == 128
    assert payload["rows"][0]["case"]["source"] == source[0]["case"]["source"]
    assert payload["rows"][0]["error_type"] == "KeyprintError"
    assert '<script>bad()' not in html
    assert "Requested format followed" in html and "format_pass:val(r.format)" in html
    assert "Boolean(r.format)" in html
    assert "language:'',claims:'',format:'',notes:''" in html
    assert "CC-BY-SA-3.0" in html and "Original source passage" in html
    assert payload["rows"][0]["format_limit"] == "Brief bullets"


@pytest.mark.parametrize("change", ["missing", "duplicate", "condition", "ratings", "rubric"])
def test_partial_unblinded_or_prejudged_review_rejected(change):
    source = rows()
    if change == "missing": source.pop()
    elif change == "duplicate": source[0]["review_id"] = source[1]["review_id"]
    elif change == "condition": source[0]["condition"] = "marked"
    elif change == "ratings": source[0]["rating"] = "pass"
    else: source[0]["rubric"]["id"] = "other"
    with pytest.raises(ValueError): build(json.dumps(source))


def test_template_changes_cannot_silently_drop_a_review_control():
    with pytest.raises(ValueError): replace_once("missing", "absent", "new")
    with pytest.raises(ValueError): replace_once("duplicate duplicate", "duplicate", "new")
