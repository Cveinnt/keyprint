import itertools
import math
from pathlib import Path

import numpy as np
import pytest
from keyprint._engine.legacy._impl.research.grouped_canonical_prototype import Profile, Config


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "tools"))
    import layer_weighted_bet
    import residual_bet
    return layer_weighted_bet, residual_bet


class Pattern(Profile):
    def __init__(self, pattern):
        super().__init__([None, b"a", b"b", b" a"], tokenizer_identity="test",
                         config=Config(history=1, layers=2, max_steps=8))
        object.__setattr__(self, "pattern", pattern)

    def bits(self, key, context, label):
        offset = (4 if context else 0) + (0 if label == b"a" else 2)
        return self.pattern[offset:offset + 2]


def test_exhaustive_ideal_null_normalization_for_each_stake_and_mixture(modules):
    weighted, residual = modules
    evidence = []
    for pattern in itertools.product((0, 1), repeat=8):
        # Two fresh contexts; predictions may be wrong about this fixed text.
        base = residual.score(Pattern(pattern), bytes(32), {"token_ids": [1, 2], "heads": [
            {"support": [0, 1, 2], "probabilities": [.1, .01, .89]},
            {"support": [1, 2, 3], "probabilities": [.9, .05, .05]}]})
        value = weighted.aggregate(base["terms"], layers=2)
        evidence.append([*map(math.exp, value["component_log_evidence"]), math.exp(value["working_log_evidence"])])
    np.testing.assert_allclose(np.mean(evidence, axis=0), np.ones(6), atol=1e-14, rtol=0)


def test_direct_products_match_whole_document_mixture(modules):
    weighted, _ = modules
    rows = [[1., -.5, .25], [-1., 0., .75]]
    result = weighted.aggregate([{"reason": "scored", "residuals": row} for row in rows], layers=3)
    # Written independently: weights at 3 layers are exactly 1, .55, .1.
    values = [math.prod(1 + stake * w * r for row in rows for w, r in zip((1., .55, .1), row))
              for stake in (1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2)]
    assert math.exp(result["working_log_evidence"]) == pytest.approx(math.fsum(values) / 5, rel=1e-14)
    assert result["working_log_evidence"] < max(result["component_log_evidence"])


def test_published_relative_weights_and_empty_evidence(modules):
    weighted, _ = modules
    np.testing.assert_allclose(weighted.weights(30), np.linspace(10, 1, 30) / 10, rtol=1e-15)
    result = weighted.aggregate([{"reason": "repeated_context", "residuals": []}])
    assert result["working_log_evidence"] == 0 and result["flagged"] is False
    assert result["calibrated"] is False


@pytest.mark.parametrize("reason,values", [
    ("scored", []), ("scored", [0.] * 29), ("scored", [0.] * 31),
    ("repeated_context", [0.] * 30), ("unknown", []),
    ("scored", [float("nan")] * 30), ("scored", [float("inf")] * 30),
    ("scored", [1.01] * 30), ("scored", [-1.01] * 30), ("scored", [True] * 30),
])
def test_malformed_or_ineligible_evidence_rejected(modules, reason, values):
    with pytest.raises(ValueError):
        modules[0].aggregate([{"reason": reason, "residuals": values}])


@pytest.mark.parametrize("layers", [1, 257, True, 30.5])
def test_invalid_layer_counts_rejected(modules, layers):
    with pytest.raises(ValueError):
        modules[0].weights(layers)


@pytest.mark.parametrize("value", [-10000., -700., -1., 0., 1., 700., 10000.])
def test_oracle_half_mixture_remains_finite_without_flooring(modules, value):
    from decimal import Decimal, localcontext
    from diagnose_surrogate_gap import half_mixture
    with localcontext() as context:
        context.prec = 60
        reference = ((Decimal(value).exp() + 1) / 2).ln()
    assert half_mixture(value) == pytest.approx(float(reference), abs=2e-12)


@pytest.mark.parametrize("value", [float("inf"), float("-inf"), float("nan")])
def test_nonfinite_oracle_ratios_rejected(modules, value):
    from diagnose_surrogate_gap import half_mixture
    with pytest.raises(ValueError):
        half_mixture(value)
