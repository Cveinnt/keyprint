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
    spec = importlib.util.spec_from_file_location("residual_bet_test", tools / "residual_bet.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Pattern(Profile):
    def __init__(self, pattern):
        super().__init__([None, b"a", b"b", b" a"], tokenizer_identity="test",
                         config=Config(history=1, layers=2, max_steps=8))
        object.__setattr__(self, "pattern", pattern)

    def bits(self, key, context, label):
        offset = 4 if context else 0
        offset += 0 if label == b"a" else 2
        return self.pattern[offset:offset + 2]


def measurement(ids, probabilities):
    return {"token_ids": ids, "heads": [
        {"support": [i for i, p in enumerate(row) if p], "probabilities": [p for p in row if p]}
        for row in probabilities]}


def test_exhaustive_two_context_null_mean_is_one_even_with_misspecified_predictions(detector):
    values = []
    for pattern in itertools.product((0, 1), repeat=8):
        # The observed text stays fixed while all ideal PRF assignments vary.
        result = detector.score(Pattern(pattern), bytes(32), measurement(
            [1, 2], [[.1, .01, .89, 0.], [.0, .9, .05, .05]]))
        values.append(math.exp(result["working_log_evidence"]))
    assert np.mean(values) == pytest.approx(1., abs=1e-14)


def test_probability_mass_on_same_canonical_label_has_zero_residual(detector):
    for pattern in itertools.product((0, 1), repeat=4):
        value = detector.score(Pattern(pattern * 2), bytes(32), measurement([3], [[.2, .1, 0., .7]]))
        assert value["terms"][0]["residuals"] == [0., 0.]
        assert value["working_log_evidence"] == 0. and not value["flagged"]


def test_repeats_and_unsupported_observations_cannot_add_fresh_evidence(detector):
    result = detector.score(Pattern((1, 1, 0, 0) * 2), bytes(32),
        measurement([0, 1, 2, 1, 2], [[0., 1., 0., 0.]] * 5))
    assert [t["reason"] for t in result["terms"]] == ["excluded_label", "scored",
        "outside_surrogate_support", "scored", "repeated_context"]
    assert result["working_log_evidence"] == 0.


def test_whole_document_mixture_is_not_best_stake_selection(detector):
    terms = [{"residuals": [.5, -.5, .5]}]
    components, result = detector.aggregate(terms)
    independent = math.fsum(math.prod(1 + s * v for v in terms[0]["residuals"])
                            for s in detector.STAKES) / len(detector.STAKES)
    assert math.exp(result) == pytest.approx(independent, rel=1e-14)
    assert result < max(components)


@pytest.mark.parametrize("ids,rows,key", [
    ([1], [[0., 1., 0., 0.]], b"bad"), ([], [], bytes(32)),
    ([1], [], bytes(32)), ([True], [[0., 1., 0., 0.]], bytes(32)),
    ([1], [[0., .9, 0., 0.]], bytes(32)), ([1], [[0., float('nan'), 0., 0.]], bytes(32))])
def test_invalid_measurements_rejected(detector, ids, rows, key):
    with pytest.raises(ValueError):
        detector.score(Pattern((1, 1, 0, 0) * 2), key, measurement(ids, rows))
