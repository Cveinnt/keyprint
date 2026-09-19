import importlib.util
import itertools
from pathlib import Path
import numpy as np
import pytest
from scipy.stats import binom


@pytest.fixture
def null():
    path = Path(__file__).parents[1] / "tools/weighted_null.py"
    spec = importlib.util.spec_from_file_location("weighted_null", path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("events,weights", [(1, (1, 2, 4)), (2, (2, 3, 5)), (3, (1, 2))])
def test_fft_matches_exhaustive_signed_bit_enumeration(null, events, weights):
    sums = np.array([sum(w*s for w,s in zip(weights * events, signs))
                     for signs in itertools.product([-1, 1], repeat=len(weights)*events)])
    tails, audit = null.lattice(events, weights)
    assert audit['alias_bound'] == 0
    for cut in range(1, max(sums) + 1):
        assert tails[cut] == pytest.approx(np.mean(sums >= cut), abs=1e-13)


def test_many_equal_weights_match_independent_binomial(null):
    tails, audit = null.lattice(100, (1,)*30)
    for successes in [1501, 1520, 1550, 1600]:
        cut = 2*successes - 3000
        assert tails[cut] == pytest.approx(binom.sf(successes-1, 3000, .5), abs=1e-12)


def test_actual_weights_are_grid_stable_at_decision_boundary(null):
    events = 300
    variance = events * sum(w*w for w in null.WEIGHTS)
    cut = int(2.58 * np.sqrt(variance))
    a, audit = null.tail_from_sum(cut, events)
    b, _ = null.tail_from_sum(cut, events, double_grid=True)
    assert abs(a - b) < 1e-11
    assert audit['alias_bound'] <= null.ALIAS_TARGET
    assert .004 < a < .006
    assert null.two_key_tail([a, .5]) == 2*a


def test_integer_weights_preserve_the_original_linear_statistic(null):
    bits = np.random.default_rng(17).binomial(1, .5, (50, 30))
    original = np.linspace(10, 1, 30)
    z = ((bits-.5)*original).sum() / (.5*np.sqrt(len(bits)*(original**2).sum()))
    measured = null.reference_tail(bits)
    from_integers = measured['centered_sum'] / np.sqrt(len(bits)*sum(w*w for w in null.WEIGHTS))
    assert from_integers == pytest.approx(z, abs=1e-13)


def test_malformed_or_uninformative_inputs_cannot_look_significant(null):
    assert null.reference_tail(np.zeros((3,30)))['reference_tail'] == 1
    for bits in [[], [[1, 0]], np.full((2,30), float('nan'))]:
        with pytest.raises(ValueError): null.reference_tail(bits)
    with pytest.raises(ValueError): null.two_key_tail([0.1])
    with pytest.raises(ValueError): null.tail_from_sum(999999, 1)
