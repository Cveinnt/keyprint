from pathlib import Path
import sys
import copy

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "tools"))
from diagnose_source_sampling import aggregate, distribution_metrics
from summarize_source_sampling import summarize


def cohort():
    rows, replay, traces = [], [], {}
    for case in range(16):
        for key in range(4):
            for arm in ("ordinary", "marked"):
                rid = f"{case}-{key}-{arm}"
                row = dict(review_id=rid, case=str(case), key_slot=key, condition=arm, tokens=2, model_calls=2)
                p = np.array([.75, .25])
                q = p if arm == "ordinary" else np.array([.25, .75])
                steps = [distribution_metrics(p, q, token, {1}) for token in (0, 1)]
                rows.append(row)
                traces[rid] = steps
                replay.append(dict(row, matched_steps=2, passed=True, diagnostics=aggregate(steps)))
    return rows, replay, traces


def test_full_cohort_retains_denominators_and_reports_eos_and_distortion():
    out = summarize(*cohort())
    ordinary, marked = out['groups']['ordinary'], out['groups']['marked']
    assert ordinary['attempts'] == marked['attempts'] == 64
    assert marked['token_weighted']['steps'] == 128
    assert ordinary['token_weighted']['mean']['total_variation'] == 0
    assert marked['token_weighted']['mean']['total_variation'] == .5
    assert marked['ending_tokens']['count'] == 64
    assert marked['eos_mass_max_absolute_change'] == .5
    assert not out['quality_acceptance'] and not out['launch_ready']


@pytest.mark.parametrize('change', ['missing', 'failed', 'reordered', 'duplicate', 'edited_trace', 'early_eos', 'missing_trace'])
def test_partial_failed_or_tampered_cohorts_never_become_summary(change):
    rows, replay, traces = cohort()
    rid = rows[0]['review_id']
    if change == 'missing': replay.pop()
    elif change == 'failed': replay[0]['passed'] = False
    elif change == 'reordered': replay.reverse()
    elif change == 'duplicate': rows[1] = copy.deepcopy(rows[0])
    elif change == 'edited_trace': traces[rid][0]['base_max'] = .9
    elif change == 'early_eos': traces[rid][0]['selected_is_eos'] = True
    elif change == 'missing_trace': del traces[rid]
    with pytest.raises(ValueError): summarize(rows, replay, traces)
