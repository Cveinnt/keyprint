import importlib.util
import itertools
from pathlib import Path
import numpy as np
import pytest


@pytest.fixture
def detector():
    path = Path(__file__).parents[1] / "tools/layer_likelihood.py"
    spec = importlib.util.spec_from_file_location("layer_likelihood", path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def test_conditional_density_normalizes_over_every_bit_pattern(detector):
    bits = np.array(list(itertools.product([0, 1], repeat=3)))
    model = {"parameters": [[1], [-2, 3], [.5, 4, -5]], "fractions": [.05, .25, 1]}
    likelihood = np.exp(detector.event_log_ratios(bits, model))
    assert likelihood.mean() == pytest.approx(1, abs=1e-14)
    # Exhaustive two-event documents: mixtures remain normalized despite
    # within-event layer dependence and adaptive conditional probabilities.
    document_values = [np.exp(detector.log_evidence(np.array([a, b]), model)) for a in bits for b in bits]
    assert np.mean(document_values) == pytest.approx(1, abs=1e-14)


def test_markov_tail_bound_on_exhaustive_null_documents(detector):
    bits = np.array(list(itertools.product([0, 1], repeat=4)))
    model = {"parameters": [[10]] + [[10] + [0] * i for i in range(1, 4)], "fractions": [.1, .5, 1]}
    values = np.array([np.exp(detector.log_evidence(np.array([a, b]), model)) for a in bits for b in bits])
    for cutoff in [2, 4, 10, 20]:
        assert np.mean(values >= cutoff) <= 1 / cutoff


def test_fit_has_no_future_layer_features_and_learns_skew(detector):
    bits = np.random.default_rng(1).binomial(1, .7, (1000, 4))
    model = detector.fit(bits)
    assert [len(p) for p in model["parameters"]] == [1, 2, 3, 4]
    assert detector.log_evidence(np.ones((30, 4)), model) > detector.log_evidence(np.zeros((30, 4)), model)


@pytest.mark.parametrize("bad", [[], [[1, 2]], [[1, float('nan')]], [1, 0]])
def test_invalid_events_are_not_silently_scored(detector, bad):
    with pytest.raises(ValueError):
        detector.event_log_ratios(bad, {"parameters": [[0]], "fractions": [1]})


def test_malformed_density_is_rejected(detector):
    with pytest.raises(ValueError):
        detector.log_evidence([[1, 0]], {"parameters": [[0], [0]], "fractions": [1]})
    with pytest.raises(ValueError):
        detector.log_evidence([[1]], {"parameters": [[0]], "fractions": [0]})
