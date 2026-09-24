import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('multikey_pilot', Path(__file__).parents[1] / 'tools/multikey_prose_pilot.py')
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)

PLAN = {'keys': 2, 'repeats': 2, 'cases': [{'id': 'french'}]}


def rows():
    return [dict(key_slot=k, repetition=r, case='french', condition=c,
                 review_id=f'{k}-{r}-{c}', completion='eos', text=f'text-key-{k}')
            for k in range(2) for r in range(2) for c in ('ordinary', 'marked')]


def test_repetition_is_grouped_by_key_and_condition():
    result = pilot.repetition_summary(PLAN, rows())
    assert result['complete']
    assert len(result['groups']) == 4
    assert all(g['distinct_texts'] == 1 and g['largest_identical_group'] == 2 for g in result['groups'])


def test_missing_failed_and_capped_attempts_never_count_as_completed_text():
    data = rows()[:-1]
    data[0]['error_type'] = 'RuntimeError'
    data[1]['completion'] = 'length'
    result = pilot.repetition_summary(PLAN, data)
    assert not result['complete']
    assert sum(g['completed'] for g in result['groups']) == 5


@pytest.mark.parametrize('change', ['duplicate', 'unknown_key', 'duplicate_id'])
def test_invalid_attempt_mapping_rejected(change):
    data = rows()
    if change == 'duplicate':
        data.append(data[0].copy())
    elif change == 'unknown_key':
        data[0]['key_slot'] = 8
    else:
        data[0]['review_id'] = data[1]['review_id']
    with pytest.raises(ValueError):
        pilot.repetition_summary(PLAN, data)
