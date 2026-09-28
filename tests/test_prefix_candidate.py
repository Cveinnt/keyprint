import importlib.util
import itertools
from pathlib import Path

import numpy as np
import pytest

from keyprint._engine.legacy._impl.research.grouped_canonical_prototype import Config, Profile
from keyprint._engine.legacy._impl.research.token_source_sparse_execution import SparseTokenSourceSession, bit_table
from keyprint._engine.legacy._impl.research.byte_trie_numeric import update

spec = importlib.util.spec_from_file_location('prefix_candidate', Path(__file__).parents[1] / 'tools/prefix_candidate.py')
prefix = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prefix)


def fixture():
    return Profile([None, b'a', b'b', b'c'], tokenizer_identity='prefix-fixture',
                   config=Config(history=1, max_steps=8))


def test_exact_five_stage_oracle_preserves_protected_and_subnormal_support():
    profile = fixture()
    q = np.array([.1, .3, .6, np.nextafter(0., 1.)])
    key = bytes(range(32))
    counters = {'branch_roundups': 0, 'partition_roundups': 0}
    actual = prefix.transform(q, profile, key, (b'x',), (), counters)
    table = bit_table(profile, key, (b'x',), [b'a', b'b', b'c'])
    expected = tuple(q[1:] / .9)
    for layer in range(5):
        expected = update(expected, [table[c][layer] for c in (b'a', b'b', b'c')])
    expected = np.maximum(np.nextafter(0., 1.), .9 * np.array(expected))
    assert actual[0] == q[0]
    assert actual[1:].tobytes() == expected.tobytes()
    protected = prefix.transform(q, profile, key, (b'x',), (2,), counters)
    assert protected[2] == q[2]
    assert np.array_equal(protected > 0, q > 0)


@pytest.mark.parametrize('condition', ['ordinary', 'marked'])
def test_session_startup_repeat_and_pending_rules(condition):
    sessions = [c(fixture(), bytes(range(32)), condition=condition)
                for c in (SparseTokenSourceSession, prefix.PrefixSession)]
    q = np.array([.1, .2, .3, .4])
    try:
        for index in range(5):
            results = [s.prepare(q) for s in sessions]
            result = results[1]
            assert not result.probabilities.flags.writeable
            assert np.array_equal(result.probabilities > 0, q > 0)
            assert result.probabilities[0] == q[0]
            if condition == 'ordinary' or index >= 2:
                assert result.probabilities.tobytes() == q.tobytes()
            with pytest.raises(RuntimeError, match='pending'):
                sessions[1].prepare(q)
            for s, r in zip(sessions, results):
                s.commit(r, 1)
    finally:
        for s in sessions:
            s.close()


def test_binomial_tail_matches_exhaustive_fair_bit_enumeration():
    for ones in range(6):
        row = np.zeros((1, 30), dtype=int)
        row[0, :ones] = 1
        exact = sum(sum(draw) >= ones for draw in itertools.product((0, 1), repeat=5)) / 32
        assert prefix.prefix_tail(row)['reference_tail'] == pytest.approx(exact)
    changed = np.zeros((1, 30), dtype=int)
    changed[:, 5:] = 1
    assert prefix.prefix_tail(changed)['reference_tail'] == 1.


def test_tournament_preserves_mean_over_exhaustive_fresh_layer_bits():
    # Finite toy oracle for the kernel argument, not a production semantic proof.
    q = (.1, .3, .6)
    outputs = []
    for bits in itertools.product((0, 1), repeat=6):
        outputs.append(update(update(q, bits[:3]), bits[3:]))
    assert np.allclose(np.mean(outputs, axis=0), q, rtol=0, atol=1e-15)


@pytest.mark.parametrize('bits', [np.zeros((0, 30)), np.zeros((3, 5)),
                                 np.full((1, 30), .5), np.full((1, 30), np.nan)])
def test_detector_rejects_missing_or_nonbinary_evidence(bits):
    with pytest.raises(ValueError, match='binary events'):
        prefix.prefix_tail(bits)


def test_detector_upper_tail_does_not_underflow_into_zero():
    result = prefix.prefix_tail(np.ones((2048, 30), dtype=int))
    assert result['reference_tail'] > 0
    assert result['trials'] == 10240


def test_candidate_has_distinct_identity_and_ordinary_parity():
    from keyprint._engine.research.keyprint_candidate_v3.adapter import Candidate
    from keyprint.backends.mlx_bounded import BoundedReferenceCandidate
    import random
    base = Candidate()
    candidates = (BoundedReferenceCandidate(base), prefix.PrefixCandidate(base))
    pipes = [c.pipeline(bytes(range(32)), condition='ordinary') for c in candidates]
    try:
        rng = [random.Random(7), random.Random(7)]
        raw = np.full((1, 151936), -np.inf, dtype=np.float32)
        raw[0, [32, 33, 34]] = 0.
        for _ in range(3):
            steps = [p.step(raw, r.getrandbits) for p, r in zip(pipes, rng)]
            assert steps[0].token_id == steps[1].token_id
        receipts = [p.receipt() for p in pipes]
        assert receipts[0]['sampling_records'] == receipts[1]['sampling_records']
        assert candidates[0].identity != candidates[1].identity
    finally:
        for p in pipes:
            p.close()
