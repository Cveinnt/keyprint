import importlib.util
import itertools
import math
from pathlib import Path
import numpy as np
import pytest

from keyprint._engine.legacy._impl.research.grouped_canonical_prototype import Profile, Config
from keyprint._engine.legacy._impl.research.token_source_policy import transform

spec = importlib.util.spec_from_file_location("log_tournament", Path(__file__).parents[1]/"tools/log_tournament.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class Pattern(Profile):
    def __init__(self, bits, layers=2):
        super().__init__([None, b"a", b"b", b" a"], tokenizer_identity="test",
                         config=Config(layers=layers))
        object.__setattr__(self, "pattern", bits)

    def bits(self, key, context, label):
        return self.pattern[:self.config.layers] if label == b"a" else self.pattern[self.config.layers:]


def test_every_small_tournament_matches_reference_and_normalizes():
    p = np.array([.1, .2, .3, .4])
    factors = [[], [], [], []]
    for bits in itertools.product((0, 1), repeat=4):
        profile = Pattern(bits)
        counters = {"branch_roundups": 0, "partition_roundups": 0}
        q = transform(p, profile, bytes(32), (), (), counters)
        assert not any(counters.values())
        for token in range(4):
            ratio = module.token_log_ratio(p, profile, bytes(32), (), token)
            assert ratio == pytest.approx(math.log(q[token] / p[token]), abs=2e-14)
            factors[token].append(math.exp(module.mixture_log_factor(ratio)))
    for values in factors:
        assert np.mean(values) == pytest.approx(1., abs=2e-14)


def test_log_space_retains_probabilities_below_float64_range():
    profile = Pattern((1,) * 30 + (0,) * 30, layers=30)
    p = np.array([0., .9, .1, 0.])
    # For the disfavored b branch, each layer squares its probability.
    ratio = module.token_log_ratio(p, profile, bytes(32), (), 2)
    expected = (2**30 - 1) * math.log(.1)
    assert ratio == pytest.approx(expected, rel=1e-14)
    assert math.isfinite(ratio) and ratio < math.log(math.ulp(0.))
    assert module.mixture_log_factor(ratio) == math.log(.5)


def test_single_label_and_excluded_tokens_contribute_no_evidence():
    p = np.array([.1, .4, 0., .5])
    for token in (0, 1, 3):
        assert module.token_log_ratio(p, Pattern((1, 0, 0, 1)), bytes(32), (), token) == 0.


@pytest.mark.parametrize("p,token", [([.5, .4, 0., 0.], 1), ([0., 1., 0., 0.], 2),
    ([0., 1., 0., 0.], True), ([0., float('nan'), 0., 0.], 1)])
def test_invalid_distribution_or_observation_rejected(p, token):
    with pytest.raises(ValueError):
        module.token_log_ratio(np.array(p), Pattern((1, 0, 0, 1)), bytes(32), (), token)
