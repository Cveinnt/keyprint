import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('prefix_summary',
    Path(__file__).parents[1] / 'tools/summarize_prefix_candidate.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture():
    rows = []
    for case in ('french', 'weather_english'):
        for arm in module.ARMS:
            rows.append(dict(key_slot=0, case=case, repetition=0, arm=arm,
                             review_id=case + arm, text=case + arm, completion='eos',
                             scores=[{'reference_tail': 1., 'prefix_five': {'reference_tail': 1.}} for _ in range(2)], matching_hit=arm != 'ordinary', other_hit=False))
    plan = dict(keys=[0], cases=[{'id': 'french'}, {'id': 'weather_english'}],
                schedule=rows.copy(), scope='test development only',
                detector={'limitations': 'not calibrated'})
    labels = [dict(review_id=r['review_id'], task_pass=True, language_pass=True, reason='fixture') for r in rows]
    return plan, rows, labels


def test_even_perfect_development_results_never_authorize_promotion():
    result = module.summarize(*fixture())
    assert result['development_screen'] == 'larger_fresh_study_required'
    assert not result['sdk_promotion'] and not result['quality_acceptance'] and not result['detector_calibrated']
    assert result['paired_reference_prefix'] == {'both_pass': 2}


@pytest.mark.parametrize('failure', ['quality', 'signal', 'inference'])
def test_different_failure_modes_reject_candidate(failure):
    plan, rows, labels = fixture()
    if failure == 'quality':
        labels[-1]['task_pass'] = False
    elif failure == 'signal':
        rows[-1]['matching_hit'] = False
    else:
        rows[-1]['error_type'] = 'RuntimeError'
    result = module.summarize(plan, rows, labels)
    assert result['development_screen'] == 'rejected'
    assert result['candidate_rejection_reasons']
    assert result['groups']['prefix_five']['attempts'] == 2


@pytest.mark.parametrize('missing', ['attempt', 'rating', 'duplicate'])
def test_incomplete_or_duplicate_evidence_is_not_summarized(missing):
    plan, rows, labels = fixture()
    if missing == 'attempt':
        rows.pop()
    elif missing == 'rating':
        labels.pop()
    else:
        rows.append(rows[0])
    with pytest.raises(ValueError):
        module.summarize(plan, rows, labels)


@pytest.mark.parametrize('condition', ['ordinary', 'wrong_key'])
def test_additional_false_hits_reject_candidate(condition):
    plan, rows, labels = fixture()
    row = next(r for r in rows if r['arm'] == ('ordinary' if condition == 'ordinary' else 'prefix_five'))
    row['scores'][0 if condition == 'ordinary' else 1]['prefix_five']['reference_tail'] = .001
    result = module.summarize(plan, rows, labels)
    assert result['development_screen'] == 'rejected'
    assert 'more ordinary or wrong-key hits under candidate detector' in result['candidate_rejection_reasons']


def test_cross_scores_do_not_choose_the_better_detector_afterward():
    plan, rows, labels = fixture()
    row = next(r for r in rows if r['case'] == 'weather_english' and r['arm'] == 'prefix_five')
    row['scores'][0]['reference_tail'] = .001
    row['matching_hit'] = False
    result = module.summarize(plan, rows, labels)
    scores = {r['detector']: r for r in result['cross_scores'] if r['arm'] == 'prefix_five'}
    assert scores['reference']['matching_hits'] == 1
    assert scores['prefix_five']['matching_hits'] == 0
    assert result['development_screen'] == 'rejected'
