"""Bitwise arithmetic and exact draw parity with the preserved dense law."""
import random

import numpy as np
import pytest

from keyprint.sampling import sparse_sample, sparse_softmax
from keyprint._engine.research.keyprint_exact_categorical_v2 import (
    SupportLossError, sample_float_weights, supported_softmax,
)
from keyprint._engine.legacy._impl.research.grouped_canonical_prototype import Config, Profile
from keyprint._engine.legacy._impl.research.token_source_sparse_execution import SparseTokenSourceSession


@pytest.mark.parametrize("width,count", [(1, 1), (31, 31), (49152, 100), (151669, 100)])
def test_bitwise_probabilities_and_draw_transcripts(width, count):
    rng = np.random.default_rng(42)
    for iteration in range(3):
        logits = np.full(width, -np.inf, dtype=np.float64)
        support = rng.choice(width, count, replace=False)
        logits[support] = rng.uniform(-600, 0, count)
        # Exercise tied weights as well as almost the full supported gap.
        if iteration == 1:
            logits[support] = 0.
        before = logits.tobytes()
        dense = np.array(supported_softmax(tuple(map(float, logits))))
        sparse = sparse_softmax(logits)
        assert dense.tobytes() == sparse.tobytes()
        assert logits.tobytes() == before
        for seed in range(3):
            expected = sample_float_weights(tuple(map(float, dense)), random.Random(seed).getrandbits)
            actual = sparse_sample(sparse, random.Random(seed).getrandbits)
            assert actual == expected


def test_marked_weights_preserve_selection_and_rejection_transcript():
    profile = Profile([None, *(f"word{i}".encode() for i in range(128))],
                      tokenizer_identity="sampling-parity-fixture", config=Config(max_steps=32))
    session = SparseTokenSourceSession(profile, bytes(range(32)), condition="marked")
    try:
        rng = np.random.default_rng(71)
        dense_rng, sparse_rng = random.Random(73), random.Random(73)
        rejections = 0
        for _ in range(24):
            logits = np.full(129, -np.inf, dtype=np.float64)
            logits[rng.choice(129, 15, replace=False)] = rng.uniform(-40, 0, 15)
            step = session.prepare(sparse_softmax(logits))
            expected = sample_float_weights(tuple(map(float, step.probabilities)), dense_rng.getrandbits)
            actual = sparse_sample(step.probabilities, sparse_rng.getrandbits)
            assert actual == expected
            rejections += sum(not draw.accepted for draw in actual.transcript)
            session.commit(step, actual.token_index)
        assert rejections > 0
    finally:
        session.close()


def test_subnormal_weights_and_degenerate_support():
    weights = np.array([0., np.nextafter(0., 1.), 0., 1., -0.], dtype=np.float64)
    assert sparse_sample(weights, random.Random(5).getrandbits) == sample_float_weights(
        tuple(map(float, weights)), random.Random(5).getrandbits)
    def no_randomness(_):
        pytest.fail("one positive interval needs no random bits")
    assert sparse_sample(np.array([0., 0., .5, 0.]), no_randomness).token_index == 2


@pytest.mark.parametrize("values", [[0., float("nan")], [0., float("inf")], [-float("inf")]])
def test_invalid_logits_rejected(values):
    with pytest.raises(ValueError):
        sparse_softmax(np.array(values))


def test_finite_support_loss_is_still_rejected():
    with pytest.raises(SupportLossError):
        sparse_softmax(np.array([0., -np.inf, -1000.]))


@pytest.mark.parametrize("values", [[0., -1.], [0., float("nan")], [float("inf")], [0., 0.]])
def test_invalid_weights_rejected_before_randomness(values):
    def forbidden(_):
        pytest.fail("invalid weights must not consume randomness")
    with pytest.raises(ValueError):
        sparse_sample(np.array(values), forbidden)


@pytest.mark.parametrize("values", [np.array([]), np.zeros((2, 2)), np.zeros(3, dtype=np.float32),
                                    np.zeros(151937), [1.]])
def test_input_contract_is_explicit(values):
    for operation in (sparse_softmax, lambda x: sparse_sample(x, random.Random(0).getrandbits)):
        with pytest.raises(ValueError):
            operation(values)


def test_rejection_cap_and_invalid_random_bits_remain_errors():
    weights = np.array([0., 1., 2., 0.])
    with pytest.raises(RuntimeError, match="rejection cap"):
        sparse_sample(weights, lambda n: (1 << n) - 1, max_draws=2)
    with pytest.raises(ValueError, match="invalid integer"):
        sparse_sample(weights, lambda n: True)
