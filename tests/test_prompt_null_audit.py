import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def modules(monkeypatch):
    directory = Path(__file__).parents[1] / 'tools'
    monkeypatch.syspath_prepend(str(directory))
    result = []
    for name in ('validate_prompt_null', 'audit_prompt_null'):
        spec = importlib.util.spec_from_file_location(name + '_test', directory / (name + '.py'))
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        result.append(module)
    return result


@pytest.mark.parametrize('hits', [0, 1, 5, 500])
def test_independent_beta_quantile_reconciles_binomial_bound(modules, hits):
    runner, auditor = modules
    rows = [{'flagged': index < hits} for index in range(500)]
    result = auditor.verify_summary(rows, runner.summarize(rows))
    assert result['complete'] and result['false_hits'] == hits


@pytest.mark.parametrize('damage', ['hits', 'available', 'bound', 'promotion', 'gate', 'unit'])
def test_summary_tampering_rejected(modules, damage):
    runner, auditor = modules
    rows = [{'flagged': False} for _ in range(500)]
    summary = runner.summarize(rows)
    if damage == 'hits': summary['false_hits'] = 1
    elif damage == 'available': summary['available'] = 1000
    elif damage == 'bound': summary['iid_only_upper_97_5_percent'] /= 2
    elif damage == 'promotion': summary['deployment_calibrated'] = True
    elif damage == 'gate': summary['null_screen_passed'] = False
    else: summary['planned'] = 1000
    with pytest.raises(AssertionError):
        auditor.verify_summary(rows, summary)


def test_missing_and_failed_records_cannot_acquire_bound(modules):
    runner, auditor = modules
    for rows in ([{'flagged': False}] * 499, [{'flagged': False}] * 499 + [{'error': 'unavailable'}]):
        result = auditor.verify_summary(rows, runner.summarize(rows))
        assert not result['complete'] and result['iid_only_upper_97_5_percent'] is None


def test_partial_record_is_never_a_completed_sample(modules, tmp_path):
    auditor = modules[1]; path = tmp_path / 'rows.jsonl'
    path.write_bytes(b'{"id": 1}\n{"id":')
    assert list(auditor.records(path, tmp_path / 'summary.json')) == [{'id': 1}]


def test_follow_catches_final_append_and_rejects_changed_prefix(modules, tmp_path):
    auditor = modules[1]; path = tmp_path / 'rows.jsonl'; done = tmp_path / 'summary.json'
    path.write_text('{"id": 1}\n')
    stream = auditor.records(path, done, follow=True)
    assert next(stream) == {'id': 1}
    path.write_text('{"id": 1}\n{"id": 2}\n'); done.touch()
    assert list(stream) == [{'id': 2}]
    done.unlink(); path.write_text('{"id": 1}\n')
    stream = auditor.records(path, done, follow=True)
    assert next(stream) == {'id': 1}
    path.write_text('{"id": 9}\n'); done.touch()
    with pytest.raises(ValueError, match='prefix changed'):
        list(stream)
