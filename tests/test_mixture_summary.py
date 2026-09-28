import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('mixture_summary',
    Path(__file__).parents[1] / 'tools/summarize_mixture_candidate.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture():
    rows = []
    for case in ('french', 'weather_english'):
        for arm in module.ARMS:
            rows.append(dict(key_slot=0, case=case, repetition=0, arm=arm,
                             review_id=case + arm, text=case + arm, completion='eos',
                             scores=[{}, {}], matching_hit=arm != 'ordinary', other_hit=False))
    plan = dict(keys=[0], cases=[{'id': 'french'}, {'id': 'weather_english'}],
                schedule=rows.copy(), scope='test development only',
                detector={'limitations': 'not calibrated'})
    labels = [dict(review_id=r['review_id'], task_pass=True, language_pass=True, reason='fixture') for r in rows]
    return plan, rows, labels


def test_even_perfect_development_results_never_authorize_promotion():
    result = module.summarize(*fixture())
    assert result['development_screen'] == 'larger_fresh_study_required'
    assert not result['sdk_promotion'] and not result['quality_acceptance'] and not result['detector_calibrated']
    assert result['paired_reference_mixture'] == {'both_pass': 2}


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
    assert result['groups']['half_mixture']['attempts'] == 2


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
