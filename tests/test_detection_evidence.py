import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def renderer():
    path = Path(__file__).parents[1] / 'tools/render_detection_evidence.py'
    spec = importlib.util.spec_from_file_location('evidence_test', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def test_exported_source_cannot_end_script_or_activate_markup(renderer):
    payload = {'text': '</script><img src=x onerror=alert(1)>\u2028&', 'score': -1.5}
    script = renderer.script_data(payload)
    assert '<' not in script and '>' not in script and '&' not in script
    assert json.loads(script.removeprefix('window.KEYPRINT_EVIDENCE = ').removesuffix(';\n')) == payload
    with pytest.raises(ValueError): renderer.script_data({'score': float('nan')})


def partial_fixture(tmp_path):
    public = tmp_path / 'public'; public.mkdir()
    plan = {'planned_documents': 500, 'tasks': [{'source_index': 7}], 'fixed_log_cutoff': 5.298317366548036}
    (public / 'plan.json').write_text(json.dumps(plan))
    row = {'id': 'null-000', 'category': 'general_qa', 'source_index': 7, 'words': 101,
           'working_log_ratios': [-1., -2.], 'flagged': False, 'text': 'must not be exported'}
    raw = (json.dumps(row) + '\n').encode(); (public / 'results.jsonl').write_bytes(raw)
    audit = {'status': 'partial_integrity_pass', 'plan_sha256': hashlib.sha256((public / 'plan.json').read_bytes()).hexdigest(),
        'results_prefix_bytes': len(raw), 'results_prefix_sha256': hashlib.sha256(raw).hexdigest(),
        'totals': {'records': 1, 'available': 1}}
    path = public / 'partial.json'; path.write_text(json.dumps(audit))
    return path


def test_explicit_partial_snapshot_never_exports_final_summary_or_source_text(renderer, tmp_path):
    audit = partial_fixture(tmp_path)
    _, summary, rows, _ = renderer.null_rows(tmp_path, audit)
    assert summary is None and len(rows) == 1 and 'text' not in rows[0]
    with (tmp_path / 'public/results.jsonl').open('ab') as stream: stream.write(b'{"later":"partial')
    assert renderer.null_rows(tmp_path, audit)[2] == rows
    with pytest.raises(FileNotFoundError): renderer.null_rows(tmp_path)


@pytest.mark.parametrize('damage', ['plan', 'prefix', 'count', 'status'])
def test_changed_or_incomplete_receipts_rejected(renderer, tmp_path, damage):
    audit = partial_fixture(tmp_path)
    if damage == 'plan': (tmp_path / 'public/plan.json').write_text('{}')
    elif damage == 'prefix': (tmp_path / 'public/results.jsonl').write_text('{}\n')
    else:
        data = json.loads(audit.read_text())
        if damage == 'count': data['totals']['records'] = 500
        else: data['status'] = 'running'
        audit.write_text(json.dumps(data))
    with pytest.raises(ValueError): renderer.null_rows(tmp_path, audit)


def test_partial_count_cannot_be_relabelled_completed(renderer, tmp_path):
    audit_path = partial_fixture(tmp_path)
    audit = json.loads(audit_path.read_text()); audit['status'] = 'pass'
    summary = {'status': 'completed', 'available': 1, 'attempts': 1, 'false_hits': 0,
               'fatal': None, 'iid_only_upper_97_5_percent': .007, 'null_screen_passed': True}
    target = tmp_path / 'public/summary.json'; target.write_text(json.dumps(summary))
    audit['summary_sha256'] = hashlib.sha256(target.read_bytes()).hexdigest()
    (tmp_path / 'public/integrity.json').write_text(json.dumps(audit))
    with pytest.raises(ValueError, match='all controls'):
        renderer.null_rows(tmp_path)


def test_recovery_cannot_export_without_full_extra_audit(renderer, tmp_path):
    audit_path = partial_fixture(tmp_path)
    plan_path = tmp_path / 'public/plan.json'; plan = json.loads(plan_path.read_text())
    plan['recovery'] = {'scope': 'recovery'}; plan_path.write_text(json.dumps(plan))
    audit = json.loads(audit_path.read_text()); audit['plan_sha256'] = renderer.sha(plan_path.read_bytes())
    audit_path.write_text(json.dumps(audit))
    with pytest.raises(ValueError, match='both complete'):
        renderer.null_rows(tmp_path, audit_path)
    summary_path = tmp_path / 'public/summary.json'
    summary_path.write_text(json.dumps({'status': 'incomplete', 'available': 1, 'attempts': 1, 'false_hits': 0}))
    audit.update(status='pass', summary_sha256=renderer.sha(summary_path.read_bytes()))
    (tmp_path / 'public/integrity.json').write_text(json.dumps(audit))
    with pytest.raises(FileNotFoundError): renderer.null_rows(tmp_path)
    (tmp_path / 'public/recovery-integrity.json').write_text(json.dumps({'status':'pass','original_attempt_unchanged':True,'original_screen_passed':True,'fresh_sample':False}))
    with pytest.raises(ValueError, match='original success'): renderer.null_rows(tmp_path)
