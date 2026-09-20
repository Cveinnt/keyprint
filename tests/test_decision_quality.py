import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('decision_quality', Path(__file__).parents[1] / 'tools/validate_decision_quality.py')
quality = importlib.util.module_from_spec(spec)
spec.loader.exec_module(quality)


@pytest.mark.parametrize('text,completion,expected,correct', [
    ('{"allowed":true}', 'eos', True, True),
    ('{"allowed":false}', 'eos', False, True),
    ('{"allowed":false}', 'eos', True, False),
    ('{"allowed":true}', 'length', True, False),
    ('{"allowed":1}', 'eos', True, False),
    ('{"allowed":"false"}', 'eos', False, False),
    ('{"allowed":false,"allowed":true}', 'eos', True, False),
    ('{"allowed":true,"explanation":"yes"}', 'eos', True, False),
    ('```json\n{"allowed":true}\n```', 'eos', True, False),
    ('null', 'eos', False, False),
])
def test_exact_answer_rejects_format_truncation_and_false_positives(text, completion, expected, correct):
    assert quality.evaluate(text, completion, expected)['correct'] is correct


def test_frozen_cases_balance_answers_and_grammar_never_forces_an_answer():
    cases = json.loads(Path(__file__).parents[1].joinpath('tools/decision_quality_cases.json').read_text())
    assert len(cases) == len({c['id'] for c in cases}) == 16
    assert {c['language'] for c in cases} == {'en', 'es', 'fr', 'zh'}
    pairs = {c['pair'] for c in cases}
    assert len(pairs) == 8
    for pair in pairs:
        rows = [c for c in cases if c['pair'] == pair]
        assert len(rows) == 2 and {c['expected'] for c in rows} == {False, True}
        assert rows[0]['prompt'] != rows[1]['prompt']
    jsonschema = pytest.importorskip('jsonschema')
    for answer in [True, False]:
        jsonschema.validate({'allowed': answer}, quality.SCHEMA)


def row(condition, correct=True, **extra):
    return {'case':'one', 'repetition':0, 'condition':condition, 'checks':{'correct':correct}, **extra}


def test_missing_and_failed_attempts_cannot_be_hidden_in_successful_screen():
    cases = [{'id':'one'}]
    partial = quality.summarize(cases, [row('ordinary')], 1)
    assert not partial['complete'] and not partial['fixed_screen_pass']
    for bad in [row('marked', False), {'case':'one','repetition':0,'condition':'marked','error_type':'RuntimeError'}]:
        result = quality.summarize(cases, [row('ordinary'), bad], 1)
        assert result['complete'] and not result['fixed_screen_pass']
        assert result['paired_outcomes'] == {'ordinary_only_correct':1}
    passed = quality.summarize(cases, [row('ordinary'), row('marked')], 1)
    assert passed['fixed_screen_pass'] and passed['quality_acceptance'] is False


@pytest.mark.parametrize('rows', [[row('ordinary'), row('ordinary')], [row('marked', repetition=1)]])
def test_duplicate_or_undeclared_attempt_is_rejected(rows):
    with pytest.raises(ValueError, match='duplicate or undeclared'):
        quality.summarize([{'id':'one'}], rows, 1)
