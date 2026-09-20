import importlib.util
import itertools
import math
from pathlib import Path

import numpy as np
import pytest

from keyprint._engine.legacy._impl.research.grouped_canonical_prototype import Profile, Config


@pytest.fixture
def detector(monkeypatch):
    tools = Path(__file__).parents[1] / "tools"
    monkeypatch.syspath_prepend(str(tools))
    spec = importlib.util.spec_from_file_location("surrogate_likelihood_test", tools / "surrogate_likelihood.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Enumerated(Profile):
    def __init__(self, pattern):
        super().__init__([None, b"a", b"b", b" a"], tokenizer_identity="test",
                         config=Config(history=1, layers=2, max_steps=32))
        object.__setattr__(self, "pattern", pattern)

    def bits(self, key, context, label):
        return self.pattern[0:2] if label == b"a" else self.pattern[2:4]


@pytest.mark.parametrize("values", [[.2, .3, .5], [.999999, .0000005, .0000005]])
def test_ideal_key_mean_factor_one_with_grouped_labels(detector, values):
    # Exhaust every independent 2-label x 2-layer bit assignment. This checks
    # the ideal-key rationale on a finite example, not HMAC deployment rates.
    factors = [[] for _ in values]
    for pattern in itertools.product((0, 1), repeat=4):
        profile = Enumerated(pattern)
        for token in (1, 2, 3):
            result = detector.score(profile, bytes(32), {"token_ids": [token],
                "heads": [{"support": [1, 2, 3], "probabilities": values}]})
            factors[token - 1].append(math.exp(result["working_log_ratio"]))
    for samples in factors:
        assert np.mean(samples) == pytest.approx(1., abs=1e-12)


def test_point_mass_contains_no_marking_information(detector):
    for pattern in itertools.product((0, 1), repeat=4):
        result = detector.score(Enumerated(pattern), bytes(32), {"token_ids": [1],
            "heads": [{"support": [1], "probabilities": [1.]}]})
        assert result["working_log_ratio"] == 0 and not result["flagged"]


def test_exclusions_zero_support_and_repeats_advance_context_honestly(detector):
    profile = Enumerated((1, 1, 0, 0))
    result = detector.score(profile, bytes(32), {"token_ids": [0, 1, 2, 1, 2],
        "heads": [{"support": [1], "probabilities": [1.]}] * 5})
    assert [t["reason"] for t in result["terms"]] == ["excluded_label", "scored",
        "outside_surrogate_support", "scored", "repeated_context"]
    assert result["working_log_ratio"] == 0


@pytest.mark.parametrize("head", [
    {"support": [1, 1], "probabilities": [.5, .5]},
    {"support": [2, 1], "probabilities": [.5, .5]},
    {"support": [True], "probabilities": [1.]},
    {"support": [4], "probabilities": [1.]},
    {"support": [1], "probabilities": [.9]},
    {"support": [1], "probabilities": [float("nan")]},
    {"support": [1], "probabilities": [0.]},
])
def test_malformed_measurements_fail_closed(detector, head):
    with pytest.raises(ValueError):
        detector.score(Enumerated((1, 1, 0, 0)), bytes(32), {"token_ids": [1], "heads": [head]})


def test_misalignment_and_bad_key_rejected(detector):
    with pytest.raises(ValueError, match="alignment"):
        detector.score(Enumerated((1, 1, 0, 0)), bytes(32), {"token_ids": [1], "heads": []})
    with pytest.raises(ValueError, match="key"):
        detector.score(Enumerated((1, 1, 0, 0)), b"bad", {})


def test_underflowing_alternative_is_finite_without_flooring(detector):
    result = detector.score(Enumerated((1, 1, 0, 0)), bytes(32), {"token_ids": [2],
        "heads": [{"support": [1, 2], "probabilities": [1., 1e-300]}]})
    assert result["working_log_ratio"] == math.log(.5)
    assert result["terms"][0]["marked_log_probability"] < math.log(math.ulp(0.))
