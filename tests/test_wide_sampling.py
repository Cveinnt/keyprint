"""Wider heads must preserve the reference arithmetic and original token IDs."""
import random

import numpy as np
import pytest

from keyprint.experimental.wide_sampling import support_filter, softmax, sample
from keyprint.sampling import sparse_softmax
from keyprint._engine.research.keyprint_stable_support_filter_v3 import stable_support_filter
from keyprint._engine.research.keyprint_exact_categorical_v2 import sample_float_weights, supported_softmax


@pytest.mark.parametrize("temperature", [float(np.nextafter(0., 1.)), 1e-35, .7, 1., 1e308])
@pytest.mark.parametrize("top_k", [1, 7, 100])
def test_filter_matches_original_within_original_domain(temperature, top_k):
    rng = np.random.default_rng(170)
    raw = rng.uniform(-1000, 1000, (1, 151936)).astype(np.float32)
    raw[:, ::17] = -np.inf
    raw[:, 55:80] = 1000  # tied cutoff; lowest original IDs must win
    before = raw.tobytes()
    a = stable_support_filter(raw, temperature=temperature, top_k=top_k, mapped_vocabulary_size=151669)
    b = support_filter(raw, temperature=temperature, top_k=top_k, mapped_vocabulary_size=151669)
    assert a.filtered_logits.tobytes() == b.filtered_logits.tobytes()
    assert a.admitted_token_ids == b.admitted_token_ids
    assert raw.tobytes() == before
    with pytest.raises(ValueError):
        b.filtered_logits.flags.writeable = True


@pytest.mark.parametrize("temperature", [float(np.nextafter(0., 1.)), 1e-35, .7, 1., 1e308])
def test_wide_head_matches_independent_full_width_definition(temperature):
    # Independent gap-first oracle: no projection or reference filter call.
    rng = np.random.default_rng(82)
    raw = rng.uniform(-1000, 0, (1, 248320)).astype(np.float32)
    raw[0, [0, 3, 151940, 248319]] = 0
    raw[0, 200000:200003] = [-600, np.nextafter(np.float32(-600), np.float32(0)),
                             np.nextafter(np.float32(-600), np.float32(-np.inf))]
    raw[0, 57] = -np.inf
    row = raw[0].astype(np.float64)
    with np.errstate(over="ignore"):
        scaled = (np.max(row) - row) / temperature
    valid = [int(i) for i in np.flatnonzero(np.isfinite(scaled) & (scaled <= 600))]
    selected = sorted(valid, key=lambda i: (-float(raw[0, i]), i))[:100]
    expected = np.full((1, len(row)), -np.inf, dtype=np.float64)
    expected[0, selected] = -scaled[selected]
    actual = support_filter(raw, temperature=temperature, top_k=100)
    assert actual.filtered_logits.tobytes() == expected.tobytes()
    assert actual.admitted_token_ids == tuple(sorted(selected))


def test_gap_boundary_and_extreme_float32_losses():
    for values, temperature in [([0, -600, -600.000061, -599.999939], 1.),
                                ([np.finfo(np.float32).max, -np.finfo(np.float32).max, 0], 1e36)]:
        raw = np.full((1, 248320), -np.inf, np.float32)
        ids = [0, 200000, 200001, 248319][:len(values)]
        raw[0, ids] = values
        original = stable_support_filter(np.array([values], np.float32), temperature=temperature, top_k=100)
        actual = support_filter(raw, temperature=temperature, top_k=100)
        assert actual.filtered_logits[0, ids].tobytes() == original.filtered_logits[0].tobytes()


def test_wide_probabilities_and_rng_transcripts_match_dense_law():
    logits = np.full(248320, -np.inf, np.float64)
    logits[[0, 151940, 248319]] = [0, -2, -600]
    actual = softmax(logits)
    expected = np.array(supported_softmax(tuple(map(float, logits))))
    assert actual.tobytes() == expected.tobytes()
    for seed in range(12):
        assert sample(actual, random.Random(seed).getrandbits) == sample_float_weights(
            tuple(map(float, expected)), random.Random(seed).getrandbits)
    weights = np.zeros(248320)
    weights[[151940, 248319]] = [np.nextafter(0., 1.), 1.]
    assert sample(weights, random.Random(4).getrandbits) == sample_float_weights(
        tuple(map(float, weights)), random.Random(4).getrandbits)
    weights[151940] = 0
    assert sample(weights, lambda _: pytest.fail("singleton needs no RNG")).token_index == 248319


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_invalid_padding_rejected_before_projection(bad):
    raw = np.zeros((1, 248320), np.float32)
    raw[0, -1] = bad
    with pytest.raises(ValueError):
        support_filter(raw, mapped_vocabulary_size=100)


@pytest.mark.parametrize("settings", [{"temperature": 0}, {"temperature": True}, {"top_k": 151670},
    {"top_k": True}, {"max_logit_gap": 601}, {"mapped_vocabulary_size": 248321}])
def test_invalid_configuration_stays_rejected(settings):
    with pytest.raises(ValueError):
        support_filter(np.zeros((1, 248320), np.float32), **settings)


def test_new_bounds_do_not_weaken_existing_contracts():
    with pytest.raises(ValueError):
        sparse_softmax(np.zeros(151937))
    with pytest.raises(ValueError):
        stable_support_filter(np.zeros((1, 248320), np.float32))
    with pytest.raises(ValueError):
        support_filter(np.zeros((1, 262145), np.float32))
    with pytest.raises(ValueError):
        softmax(np.zeros(151670))
    with pytest.raises(ValueError):
        sample(np.ones(151670), lambda _: pytest.fail("invalid support consumed RNG"))
    weights = np.zeros(248320)
    weights[[151940, 248319]] = [1., 2.]
    with pytest.raises(RuntimeError, match="rejection cap"):
        sample(weights, lambda n: (1 << n) - 1, max_draws=2)
    with pytest.raises(ValueError, match="invalid integer"):
        sample(weights, lambda _: True)
