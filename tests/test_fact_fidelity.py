import json
from pathlib import Path
import sys

import pytest

tools = Path(__file__).parents[1] / 'tools'
sys.path.insert(0, str(tools))
try:
    from summarize_fact_fidelity import summarize
    from build_fact_review import build
finally:
    sys.path.pop(0)


def fixture():
    case = {'id': 'fixture', 'facts': ['No approval yet'], 'max_words': 5}
    rows = [{'key_slot': 0, 'case': 'fixture', 'condition': c, 'review_id': c,
             'completion': 'eos', 'text': 'No approval yet.'} for c in ('ordinary', 'marked')]
    plan = {'cases': [case], 'schedule': rows.copy(), 'scope': 'fixture'}
    labels = [{'review_id': c, 'fact_checks': [True], 'no_unsupported_claims': True,
               'language_pass': True, 'reason': 'fixture'} for c in ('ordinary', 'marked')]
    return plan, rows, labels


def test_content_and_format_failures_remain_distinct():
    plan, rows, labels = fixture()
    rows[1]['text'] = 'No approval has been received so far.'
    result = summarize(plan, rows, labels)
    assert result['groups']['marked']['content_pass'] == 1
    assert result['groups']['marked']['format_pass'] == 0
    assert result['groups']['marked']['task_pass'] == 0
    assert result['paired']['content_pass'] == {'both_pass': 1}
    assert result['paired']['task_pass'] == {'ordinary_only': 1}
    assert not result['quality_acceptance'] and not result['detector_calibrated']


@pytest.mark.parametrize('field', ['fact', 'claims', 'language', 'runtime', 'cap'])
def test_each_material_failure_prevents_task_pass(field):
    plan, rows, labels = fixture()
    if field == 'fact': labels[1]['fact_checks'][0] = False
    if field == 'claims': labels[1]['no_unsupported_claims'] = False
    if field == 'language': labels[1]['language_pass'] = False
    if field == 'runtime': rows[1]['error_type'] = 'RuntimeError'
    if field == 'cap': rows[1]['completion'] = 'length'
    assert summarize(plan, rows, labels)['groups']['marked']['task_pass'] == 0


@pytest.mark.parametrize('bad', ['missing_attempt', 'duplicate_attempt', 'missing_rating', 'unknown_rating', 'fact_count', 'nonboolean'])
def test_incomplete_or_malformed_evidence_cannot_pass(bad):
    plan, rows, labels = fixture()
    if bad == 'missing_attempt': rows.pop()
    if bad == 'duplicate_attempt': rows.append(rows[0])
    if bad == 'missing_rating': labels.pop()
    if bad == 'unknown_rating': labels[0]['review_id'] = 'unknown'
    if bad == 'fact_count': labels[0]['fact_checks'] = []
    if bad == 'nonboolean': labels[0]['fact_checks'] = ['yes']
    with pytest.raises(ValueError): summarize(plan, rows, labels)


def review_row():
    return {'review_id': 'fixture', 'text': '</script><script>window.injected=1</script>',
            'completion': 'eos', 'case': {'id': 'fixture', 'language': 'English',
              'facts': ['No approval yet'], 'prompt': 'Preserve the fact', 'forbidden': 'Approval', 'max_words': 5}}


def test_review_builder_keeps_generated_markup_inert_and_ratings_empty():
    source = json.dumps([review_row()])
    output = build(source)
    assert '</script><script>window.injected=1</script>' not in output
    assert '\\u003c/script\\u003e' in output
    assert 'ratings:{}' in output
    assert '__REVIEW_DATA__' not in output


@pytest.mark.parametrize('field', ['condition', 'key_slot', 'scores', 'arm'])
def test_review_page_refuses_unblinded_metadata(field):
    row = review_row(); row[field] = 'hidden'
    with pytest.raises(ValueError, match='exposes'):
        build(json.dumps([row]))
