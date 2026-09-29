import ast
import copy
from pathlib import Path
import sys

import pytest

TOOLS = Path(__file__).parents[1]/'tools'
sys.path.insert(0, str(TOOLS))
from continue_source_sampling import validate_prefix, validate_preflight
from preflight_guarded_source import selected_rows
from multikey_prose_pilot import digest
from test_source_sampling_summary import cohort


def test_completed_prefix_returns_every_remaining_original_path():
    rows, replay, traces = cohort()
    assert validate_prefix(rows, replay[:111], traces.__getitem__) == rows[111:]


@pytest.mark.parametrize('change', ['empty', 'complete', 'failed', 'reorder', 'trace', 'count', 'early_eos'])
def test_invalid_previous_prefix_cannot_be_resumed(change):
    rows, replay, traces = cohort()
    prefix = replay[:111]
    if change == 'empty': prefix = []
    elif change == 'complete': prefix = replay
    elif change == 'failed': prefix[0]['passed'] = False
    elif change == 'reorder': prefix.reverse()
    elif change == 'trace': traces[rows[0]['review_id']][0]['base_max'] = .123
    elif change == 'count': prefix[0]['matched_steps'] -= 1
    elif change == 'early_eos': traces[rows[0]['review_id']][0]['selected_is_eos'] = True
    with pytest.raises(ValueError): validate_prefix(rows, prefix, traces.__getitem__)


def test_fixed_preflight_never_selects_by_outcome():
    rows, _, _ = cohort()
    plan = {'schedule': copy.deepcopy(rows)}
    assert selected_rows(rows, plan) == rows[:2]
    for bad in (rows[:-1], rows[::-1]):
        with pytest.raises(ValueError): selected_rows(bad, plan)


def preflight():
    names = ['preflight_guarded_source.py', 'replay_wide_mlx.py', 'audit_pydantic_ai.py',
        'multikey_prose_pilot.py', 'memory_watchdog.py', 'mlx_guarded_worker.py']
    return {'passed': True, 'matched_steps': 16,
        'rows': [{'review_id': rid, 'passed': True, 'matched_steps': 8} for rid in ['a', 'b']],
        'plan': {'scripts_sha256': {n: digest(TOOLS/n) for n in names},
            'cache_limit_bytes': 0, 'plan_sha256': 'plan', 'runs_sha256': 'runs', 'rows': ['a', 'b']}}, {
        'status': 'completed', 'exit_code': 0, 'cleanup_verified': True,
        'limit_bytes': 10*1024**3, 'peak_footprint_bytes': 7*1024**3}


def test_successful_preflight_is_bound_to_scripts_study_and_guard():
    validate_preflight(*preflight(), 'plan', 'runs', ['a', 'b'])


@pytest.mark.parametrize('change', ['failure', 'partial', 'study', 'order', 'scripts', 'hash', 'guard', 'cleanup', 'overshoot', 'cache'])
def test_unqualified_preflight_cannot_authorize_continuation(change):
    p, g = preflight()
    if change == 'failure': p['rows'][0]['passed'] = False
    elif change == 'partial': p['rows'][0]['matched_steps'] = 7
    elif change == 'study': p['plan']['runs_sha256'] = 'other'
    elif change == 'order': p['rows'].reverse()
    elif change == 'scripts': p['plan']['scripts_sha256'] = {}
    elif change == 'hash': p['plan']['scripts_sha256']['memory_watchdog.py'] = 'other'
    elif change == 'guard': g['status'] = 'stopped'
    elif change == 'cleanup': g['cleanup_verified'] = False
    elif change == 'overshoot': g['peak_footprint_bytes'] = 11*1024**3
    elif change == 'cache': p['plan']['cache_limit_bytes'] = 1
    with pytest.raises(ValueError): validate_preflight(p, g, 'plan', 'runs', ['a', 'b'])


def test_native_replay_loop_is_unchanged_from_frozen_diagnostic():
    def loop(name):
        tree = ast.parse((TOOLS/name).read_text())
        return next(ast.dump(n, include_attributes=False) for n in ast.walk(tree)
            if isinstance(n, ast.For) and isinstance(n.target, ast.Tuple)
            and [getattr(x, 'id', None) for x in n.target.elts] == ['head', 'draw', 'token'])
    assert loop('continue_source_sampling.py') == loop('diagnose_source_sampling.py')
