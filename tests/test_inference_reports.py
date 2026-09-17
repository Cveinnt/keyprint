"""Validate the real-inference harness's reporting boundaries."""
import json

from tools.validate_compatibility import screens, write_report


def test_json_screen_rejects_strings_numbers_and_wrappers():
    case = {"expected_json": {"name": "Maya", "count": 3, "enabled": False}}
    assert screens(case, '{"name":"Maya","count":3,"enabled":false}', "eos")["exact_json"]
    for text in ('{"name":"Maya","count":3,"enabled":"false"}',
                 '{"name":"Maya","count":3.0,"enabled":false}', '```json\n{}\n```', '[]'):
        assert screens(case, text, "eos")["exact_json"] is False


def test_quality_flags_do_not_approve_semantics_or_hide_failures(tmp_path):
    case = {"id":"email", "prompt":"Write an email", "review":"Check exact time", "required_literals":["09:30"]}
    row = {"case":"email", "condition":"marked", "text":"Meet at 09h30 <script>bad()</script>",
           "screens":screens(case, "Meet at 09h30", "length")}
    report = {"backend":"test", "cases":[case], "runs":[row], "clients":{}, "engineering_failures":[]}
    (tmp_path/"owner.key").write_bytes(b"private fixture")
    write_report(tmp_path, report)
    exported = json.loads((tmp_path/"public/comparison.json").read_text())
    assert len(exported["screening_failures"]) == 1
    assert exported["quality_acceptance"].startswith("not_established")
    page = (tmp_path/"public/comparison.html").read_text()
    assert "<script>bad()" not in page and "&lt;script&gt;" in page
    assert "private fixture" not in page
    assert sorted(p.name for p in (tmp_path/"public").iterdir()) == ["comparison.html", "comparison.json"]
