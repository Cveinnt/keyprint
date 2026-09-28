from pathlib import Path
import sys
import copy
import json
from fractions import Fraction

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "tools"))
from diagnose_source_sampling import aggregate, distribution_metrics
from summarize_source_sampling import summarize, main
from multikey_prose_pilot import digest


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


def test_longer_outputs_do_not_silently_dominate_equal_output_summary():
    rows, replay, traces = cohort()
    rid = rows[0]['review_id']
    traces[rid].insert(0, copy.deepcopy(traces[rid][0]))
    rows[0].update(tokens=3, model_calls=3)
    replay[0].update(matched_steps=3, model_calls=3, diagnostics=aggregate(traces[rid]))
    group = summarize(rows, replay, traces)['groups']['ordinary']
    token_mean = Fraction(64, 1) + Fraction(3,4)
    output_mean = (63 * Fraction(1,2) + Fraction(7,12)) / 64
    assert group['token_weighted']['mean']['selected_base_probability'] == pytest.approx(float(token_mean / 129))
    assert group['equal_output_mean']['selected_base_probability'] == pytest.approx(float(output_mean))
    assert float(token_mean / 129) != float(output_mean)


def files_for_main(root):
    rows, replay, traces = cohort()
    public, private = root/'public', root/'private'
    public.mkdir(); (private/'sampling-diagnosis').mkdir(parents=True)
    write = lambda f, obj: f.write_text(json.dumps(obj))
    write(public/'plan.json', {'fixture': True})
    write(private/'runs.json', rows)
    plan = {'runs_sha256': digest(private/'runs.json'),
        'study_plan_sha256': digest(public/'plan.json'),
        'script_sha256': digest(Path(__file__).parents[1]/'tools/diagnose_source_sampling.py')}
    write(public/'sampling-diagnosis-plan.json', plan)
    result = {'plan': plan, 'attempts': 128, 'verified': 128,
        'matched_steps': 256, 'rows': replay}
    write(public/'sampling-diagnosis-results.json', result)
    for rid, values in traces.items():
        write(private/'sampling-diagnosis'/(rid+'.json'), values)
    return public, private, result


def test_file_entrypoint_binds_final_result_and_all_trace_hashes(tmp_path, monkeypatch):
    public, private, result = files_for_main(tmp_path)
    monkeypatch.setattr(sys, 'argv', ['summarize_source_sampling', str(tmp_path)])
    main()
    out = json.loads((public/'sampling-diagnosis-summary.json').read_text())
    assert out['commitment']['results_sha256'] == digest(public/'sampling-diagnosis-results.json')
    assert len(out['commitment']['trace_sha256']) == 128
    for rid, sha in out['commitment']['trace_sha256'].items():
        assert sha == digest(private/'sampling-diagnosis'/(rid+'.json'))
    assert out['groups']['ordinary']['attempts'] == out['groups']['marked']['attempts'] == 64


@pytest.mark.parametrize('change', ['only_progress', 'failed', 'runs_hash', 'plan_hash', 'script_hash', 'step_count'])
def test_file_entrypoint_rejects_incomplete_or_unbound_results_without_output(tmp_path, monkeypatch, change):
    public, private, result = files_for_main(tmp_path)
    final = public/'sampling-diagnosis-results.json'
    if change == 'only_progress':
        final.rename(private/'sampling-diagnosis/progress.json')
    elif change == 'failed': result['verified'] = 127
    elif change == 'runs_hash': (private/'runs.json').write_text((private/'runs.json').read_text()+'\n')
    elif change == 'plan_hash': (public/'plan.json').write_text('{}')
    elif change == 'script_hash':
        result['plan']['script_sha256'] = 'wrong'
        (public/'sampling-diagnosis-plan.json').write_text(json.dumps(result['plan']))
    elif change == 'step_count': result['matched_steps'] = 255
    if change != 'only_progress': final.write_text(json.dumps(result))
    monkeypatch.setattr(sys, 'argv', ['summarize_source_sampling', str(tmp_path)])
    with pytest.raises((ValueError, FileNotFoundError)): main()
    assert not (public/'sampling-diagnosis-summary.json').exists()
