import importlib.util
from pathlib import Path
import random

import numpy as np
import pytest

from keyprint._engine.legacy._impl.research.grouped_canonical_prototype import Config, Profile
from keyprint._engine.legacy._impl.research.token_source_sparse_execution import SparseTokenSourceSession
from keyprint._engine.research.keyprint_exact_categorical_v2 import sample_float_weights

spec = importlib.util.spec_from_file_location('concentration_audit',
    Path(__file__).parents[1] / 'tools/audit_probability_concentration.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)
summary_spec = importlib.util.spec_from_file_location('concentration_summary',
    Path(__file__).parents[1] / 'tools/summarize_probability_concentration.py')
summary = importlib.util.module_from_spec(summary_spec)
summary_spec.loader.exec_module(summary)


def generate(condition):
    profile = Profile([None, b'a', b'b', b'c', b' '], tokenizer_identity='audit-fixture',
                      config=Config(history=1, max_steps=20))
    session = SparseTokenSourceSession(profile, bytes(range(32)), condition=condition)
    rng = random.Random(43)
    outputs = []
    try:
        for i in range(12):
            # Include excluded mass, subnormal support and repeated contexts.
            q = np.array([.1, .2, .3, .4, np.nextafter(0., 1.)], dtype=np.float64)
            before = q.tobytes()
            prepared = session.prepare(q)
            assert session._pending is prepared
            sampled = sample_float_weights(tuple(map(float, prepared.probabilities)), rng.getrandbits)
            session.commit(prepared, sampled.token_index)
            assert q.tobytes() == before
            outputs.append((prepared.probabilities.tobytes(), sampled))
        return outputs, session.source_receipt()
    finally:
        session.close()


@pytest.mark.parametrize('condition', ['ordinary', 'marked'])
def test_observer_preserves_probabilities_draws_commits_and_receipt(condition):
    expected = generate(condition)
    rows = []
    original = SparseTokenSourceSession.prepare
    with audit.observe(rows):
        actual = generate(condition)
    assert actual == expected
    assert SparseTokenSourceSession.prepare is original
    assert 'commit' not in SparseTokenSourceSession.__dict__
    assert len(rows) == 12
    assert all(r['reference_bitwise_equal'] and r['layer_replay_bitwise_equal'] for r in rows)
    assert any(r['repeated_context'] for r in rows)
    assert all(r['base']['support'] == r['prepared']['support'] == 5 for r in rows)
    if condition == 'ordinary':
        assert all(r['total_variation'] == 0 and r['kl_prepared_base_bits'] == 0 for r in rows)
    else:
        assert any(len(r['stages']) == 31 for r in rows)


def test_hooks_restored_after_observation_failure(monkeypatch):
    original = SparseTokenSourceSession.prepare
    def fail(*args):
        raise AssertionError('Diagnostic failed')
    monkeypatch.setattr(audit, 'inspect_step', fail)
    with pytest.raises(AssertionError, match='Diagnostic failed'):
        with audit.observe([]):
            generate('marked')
    assert SparseTokenSourceSession.prepare is original
    assert 'commit' not in SparseTokenSourceSession.__dict__


def test_reference_disagreement_fails_audit(monkeypatch):
    monkeypatch.setattr(audit, 'reference_transform', lambda q, *args: q.copy())
    with pytest.raises(AssertionError, match='Reference and active'):
        with audit.observe([]):
            generate('marked')


def test_metrics_have_known_limits():
    uniform = audit.metrics(np.array([.25, .25, .25, .25, 0.]))
    assert uniform == dict(entropy_bits=2., max_probability=.25, collision_probability=.25, support=4)
    point = audit.metrics(np.array([0., 1.]))
    assert point == dict(entropy_bits=0., max_probability=1., collision_probability=1., support=1)


def recorded_rows():
    rows = []
    for condition in ('ordinary', 'marked'):
        steps = []
        with audit.observe(steps):
            generate(condition)
        rows.append(dict(key_slot=0, condition=condition, steps=steps, completion='eos',
                         exact_token_trace=True, model_forward_calls=12, draw_calls=12,
                         draw_transcript_sha256=condition, matches_prior_same_condition=0))
    return rows


def test_summary_keeps_both_conditions_and_same_prefix_comparison():
    result = summary.summarize({'keys': [0]}, recorded_rows())
    assert result['attempts'] == 2
    assert result['committed_steps'] == 24
    ordinary, marked = result['rows']
    assert ordinary['all_steps']['mean_total_variation'] == 0
    assert marked['all_steps']['mean_total_variation'] > 0
    assert marked['repeated_context_steps'] > 0
    assert len(marked['layer_means']) == 31


@pytest.mark.parametrize('defect', ['missing', 'duplicate', 'error', 'uncommitted', 'parity', 'no_inference'])
def test_summary_rejects_missing_or_invalid_evidence(defect):
    rows = recorded_rows()
    if defect == 'missing':
        rows.pop()
    elif defect == 'duplicate':
        rows.append(rows[0])
    elif defect == 'error':
        rows[0]['error_type'] = 'RuntimeError'
    elif defect == 'uncommitted':
        del rows[0]['steps'][0]['token_id']
    elif defect == 'parity':
        rows[0]['steps'][0]['reference_bitwise_equal'] = False
    else:
        rows[0]['model_forward_calls'] = 0
    with pytest.raises(ValueError):
        summary.summarize({'keys': [0]}, rows)


@pytest.mark.parametrize('defect', [None, 'base_hash', 'prepared_hash', 'token', 'condition'])
def test_independent_sdk_receipts_must_match_observed_probabilities(defect):
    row = recorded_rows()[1]
    payload = dict(assigned_condition='marked',
                   committed_token_ids=[s['token_id'] for s in row['steps']],
                   sampling_records=[dict(step=s['step'], token_id=s['token_id'],
                       base_probability_sha256=s['base_sha256'],
                       prepared_probability_sha256=s['prepared_sha256']) for s in row['steps']])
    if defect is None:
        summary.check_probability_receipt(row, payload)
        return
    if defect == 'base_hash':
        payload['sampling_records'][0]['base_probability_sha256'] = 'changed'
    elif defect == 'prepared_hash':
        payload['sampling_records'][0]['prepared_probability_sha256'] = 'changed'
    elif defect == 'token':
        payload['committed_token_ids'][0] = -1
    else:
        payload['assigned_condition'] = 'ordinary'
    with pytest.raises(ValueError):
        summary.check_probability_receipt(row, payload)
