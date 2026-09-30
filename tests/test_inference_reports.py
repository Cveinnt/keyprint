"""Validate the real-inference harness's reporting boundaries."""
import json
import pytest

from tools.validate_compatibility import screens, write_report, load_cases


def test_expected_provider_rejection_does_not_approve_quality(tmp_path):
    result = {"status": "expected_rejection", "error_type": "RewriteUnavailableError"}
    report = {"backend": "fixture", "cases": [], "runs": [],
              "clients": {"openai_object_local_rewrite": result},
              "engineering_failures": ["unrelated-runtime-failure"]}
    write_report(tmp_path, report)
    saved = json.loads((tmp_path / "public/comparison.json").read_text())
    assert saved["engineering_failures"] == ["unrelated-runtime-failure"]
    assert saved["clients"]["openai_object_local_rewrite"] == result
    assert saved["quality_acceptance"].startswith("not_established")
    page = (tmp_path / "public/comparison.html").read_text()
    assert "Expected rejection" in page
    assert "not hosted-model integration" in page


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


def test_word_limit_violation_is_retained_as_a_quality_flag(tmp_path):
    case = {"id":"bounded", "prompt":"Use at most two words", "review":"No more than two", "max_tokens":8, "max_words":2}
    row = {"case":"bounded", "condition":"ordinary", "text":"one two three", "screens":screens(case,"one two three","eos")}
    report = {"backend":"fixture", "cases":[case], "runs":[row], "clients":{}, "engineering_failures":[]}
    write_report(tmp_path,report)
    exported = json.loads((tmp_path/'public/comparison.json').read_text())
    assert exported['screening_failures'] == [dict(case='bounded',condition='ordinary',screens=row['screens'])]
    assert row['screens']['whitespace_word_count'] == 3
    assert row['screens']['semantic_quality'] == 'requires_review'


@pytest.mark.parametrize('change', [dict(id='../escape'), dict(id='a/b'), dict(max_tokens=True),
    dict(max_tokens=1025), dict(prompt=' '), dict(required_literals='Maya'), dict(max_words=False),
    dict(action='unknown'), dict(action='rewrite', preserve=['absent']),
    dict(action='rewrite', prompt='```code```'), dict(action='rewrite', preserve='hello')])
def test_custom_cases_reject_unsafe_or_unbounded_requests(tmp_path,change):
    case = {'id':'safe','prompt':'hello','review':'review','max_tokens':8,**change}
    path = tmp_path/'cases.json';path.write_text(json.dumps([case]))
    with pytest.raises(ValueError):load_cases(path)


def test_custom_cases_reject_duplicate_artifact_paths(tmp_path):
    case = {'id':'same','prompt':'hello','review':'review','max_tokens':8}
    path = tmp_path/'cases.json';path.write_text(json.dumps([case,case]))
    with pytest.raises(ValueError,match='unique'):load_cases(path)


def test_rewrite_failure_is_flagged_even_when_text_screens_pass(tmp_path):
    case = {'id':'rewrite', 'prompt':'Hi Maya.', 'review':'Keep name', 'max_tokens':32,
            'action':'rewrite', 'preserve':['Maya']}
    path = tmp_path/'cases.json'; path.write_text(json.dumps([case]))
    assert load_cases(path) == [case]
    row = {'case':'rewrite', 'condition':'marked', 'text':'Maya, hello Maya.',
           'screens':{**screens(case, 'Maya, hello Maya.', 'eos'), 'rewrite_checks_passed':False}}
    report = {'backend':'fixture', 'cases':[case], 'runs':[row], 'clients':{}, 'engineering_failures':[]}
    write_report(tmp_path, report)
    exported = json.loads((tmp_path/'public/comparison.json').read_text())
    assert len(exported['screening_failures']) == 1
    assert exported['quality_acceptance'].startswith('not_established')


@pytest.mark.parametrize("instruction,candidate,literals", [
    ("Rédigez en français. Maya doit approuver.", "Maya must approve.", ["Maya"]),
    ("Do not send the file to Maya.", "Send the file to Maya.", ["Maya"]),
    ("Wait for Maya's approval, not just a reply.", "Proceed when Maya replies.", ["Maya"]),
    ("Maya approves; Ravi receives the file.", "Ravi approves; Maya receives the file.", ["Maya", "Ravi"]),
    ("Review at 09:30; publish at 14:00.", "Review at 14:00; publish at 09:30.", ["09:30", "14:00"]),
    ("Approval is pending from Maya.", "Approval was received from Maya.", ["Maya"]),
])
def test_literal_pass_never_approves_language_or_meaning_failures(tmp_path, instruction, candidate, literals):
    case = {"id":"preservation", "prompt":instruction, "review":"Preserve every obligation",
            "required_literals":literals}
    row = {"case":case["id"], "condition":"marked", "text":candidate,
           "screens":screens(case, candidate, "eos")}
    report = {"backend":"fixture", "cases":[case], "runs":[row], "clients":{}, "engineering_failures":[]}
    write_report(tmp_path, report)
    saved = json.loads((tmp_path/'public/comparison.json').read_text())
    assert saved['screening_failures'] == []  # These lexical checks miss the semantic failure.
    assert saved['runs'][0]['text'] == candidate
    verdict = saved['runs'][0]['output_quality']
    assert verdict['status'] == 'unreviewed' and verdict['approved_for_delivery'] is False
    assert saved['output_quality_summary'] == {'approved':0,'blocked':0,'unreviewed':1}
    assert 'Not approved for delivery' in (tmp_path/'public/comparison.html').read_text()


@pytest.mark.parametrize('row', [
    {'error_type':'RewriteUnavailableError'}, {},
    {'screens':{'complete':False,'nonempty':True,'missing_literals':[]}},
    {'screens':{'complete':True,'nonempty':False,'missing_literals':[]}},
    {'screens':{'complete':True,'nonempty':True,'missing_literals':['Maya']}},
])
def test_missing_or_failed_quality_evidence_is_blocked(row):
    from tools.validate_compatibility import quality_verdict
    verdict = quality_verdict(row)
    assert verdict['status'] == 'blocked'
    assert verdict['approved_for_delivery'] is False and verdict['reasons']
