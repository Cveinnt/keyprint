from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / 'tools'))
try:
    from summarize_long_fidelity import summarize
finally:
    sys.path.pop(0)


def fixture():
    case = {'id': 'fixture', 'facts': ['Status pending'], 'min_words': 6,
            'max_words': 12, 'paragraphs': 3}
    rows = [{'key_slot': 0, 'case': 'fixture', 'condition': c, 'review_id': c,
             'completion': 'eos', 'text': 'Fact retained.\n\nDetail retained.\n\nStatus pending.',
             'matching_hit': c == 'marked', 'other_hit': False}
            for c in ('ordinary', 'marked')]
    labels = [{'review_id': c, 'fact_checks': [True], 'no_unsupported_claims': True,
               'language_pass': True, 'prose_pass': True, 'reason': 'fixture'}
              for c in ('ordinary', 'marked')]
    return {'cases': [case], 'schedule': rows.copy(), 'scope': 'fixture'}, rows, labels


def test_joint_requires_same_output_to_pass_content_and_detection():
    plan, rows, labels = fixture()
    result = summarize(plan, rows, labels)
    assert result['groups']['marked']['joint_pass'] == 1
    labels[1]['fact_checks'][0] = False
    result = summarize(plan, rows, labels)
    assert result['groups']['marked']['matching_hits'] == 1
    assert result['groups']['ordinary']['content_pass'] == 1
    assert result['groups']['marked']['joint_pass'] == 0
    assert not result['quality_acceptance'] and not result['detector_calibrated']


@pytest.mark.parametrize('failure', ['too_short', 'too_long', 'paragraphs', 'not_prose', 'wrong_key', 'no_signal', 'cap'])
def test_each_joint_requirement_is_enforced(failure):
    plan, rows, labels = fixture()
    if failure == 'too_short': rows[1]['text'] = 'One.\n\nTwo.\n\nThree.'
    if failure == 'too_long': rows[1]['text'] = 'One two three four five.\n\nOne two three four five.\n\nOne two three four five.'
    if failure == 'paragraphs': rows[1]['text'] = 'All six words in one paragraph.'
    if failure == 'not_prose': labels[1]['prose_pass'] = False
    if failure == 'wrong_key': rows[1]['other_hit'] = True
    if failure == 'no_signal': rows[1]['matching_hit'] = False
    if failure == 'cap': rows[1]['completion'] = 'length'
    assert summarize(plan, rows, labels)['groups']['marked']['joint_pass'] == 0


def test_japanese_uses_character_range_not_whitespace_words():
    plan, rows, labels = fixture()
    case = plan['cases'][0]
    del case['min_words']; del case['max_words']
    case.update(min_chars=8, max_chars=10)
    rows[1]['text'] = '確認\n\n保留\n\n未定'
    result = summarize(plan, rows, labels)
    assert result['judgments']['marked']['length'] == 10
    assert result['groups']['marked']['joint_pass'] == 1


def test_missing_data_and_ambiguous_prose_rating_are_rejected():
    plan, rows, labels = fixture()
    labels[1]['prose_pass'] = None
    with pytest.raises(ValueError): summarize(plan, rows, labels)
    plan, rows, labels = fixture()
    rows.pop()
    with pytest.raises(ValueError): summarize(plan, rows, labels)
