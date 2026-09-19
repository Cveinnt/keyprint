import importlib.util
import itertools
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.stats import binom


@pytest.fixture
def detector(monkeypatch):
    for name in ("layer_likelihood", "weighted_null", "learned_layer_weights"):
        path = Path(__file__).parents[1] / "tools" / f"{name}.py"
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
    return module


def test_learned_fixed_weight_tail_matches_every_small_null_outcome(detector):
    weights = [1, 2, 5]
    outcomes = [np.array(bits).reshape(2, 3) for bits in itertools.product([0, 1], repeat=6)]
    sums = [int(((2 * b - 1) * weights).sum()) for b in outcomes]
    for bits, statistic in zip(outcomes, sums):
        measured = detector.reference_tail(bits, weights)
        expected = 1 if statistic <= 0 else np.mean(np.array(sums) >= statistic) + 1e-9
        assert measured["reference_tail"] == pytest.approx(expected, abs=1e-13)
        assert detector.reference_tail(bits, weights, double_grid=True)["reference_tail"] == pytest.approx(expected, abs=1e-13)


def test_equal_scaled_weights_match_binomial_probability(detector):
    bits = np.zeros((100, 3))
    bits.flat[:180] = 1
    assert detector.reference_tail(bits, [256] * 3)["reference_tail"] == pytest.approx(binom.sf(179, 300, .5) + 1e-9, abs=1e-12)


def test_fit_assigns_more_weight_to_stronger_training_signal(detector):
    bits = np.zeros((100, 4))
    for i, positives in enumerate((70, 60, 50, 40)):
        bits[:positives, i] = 1
    model = detector.fit_weights(bits)
    weights = model["weights_integer"]
    assert weights[0] == 256 and weights[0] > weights[1] > weights[2] == weights[3] == 1
    assert model["fit_events"] == 100
    with pytest.raises(ValueError, match="no positive"):
        detector.fit_weights(np.zeros((10, 3)))


@pytest.mark.parametrize("weights", [[], [0], [-1], [257], [1.0], [True]])
def test_invalid_weights_fail_closed(detector, weights):
    with pytest.raises(ValueError):
        detector.reference_tail([[1]], weights)


def test_layer_mismatch_and_invalid_bits_fail_closed(detector):
    with pytest.raises(ValueError):
        detector.reference_tail([[1, 0]], [1])
    with pytest.raises(ValueError):
        detector.reference_tail([[float("nan")]], [1])
    with pytest.raises(ValueError):
        detector.reference_tail(np.ones((2049, 1)), [1])
