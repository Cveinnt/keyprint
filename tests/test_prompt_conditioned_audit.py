import importlib.util
import math
from decimal import Decimal, localcontext
from pathlib import Path

import pytest

from keyprint._engine.legacy._impl.research.grouped_canonical_prototype import Profile, Config


@pytest.fixture
def audit():
    path = Path(__file__).parents[1] / "tools/audit_prompt_conditioned_likelihood.py"
    spec = importlib.util.spec_from_file_location("prompt_likelihood_audit_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("weights", [[.1,.2,.3,.4], [0.,1.,1e-300,1e-300]])
@pytest.mark.parametrize("context", [(), (b"a",)])
def test_independent_log_transform_matches_decimal_ordered_pair_enumeration(audit, weights, context):
    profile = Profile([None,b"a",b"b",b"a"], tokenizer_identity="audit-test",config=Config(history=1,layers=4,max_steps=32))
    key = bytes(range(32))
    support = [i for i,p in enumerate(weights) if p > 0]
    probabilities = [weights[i] for i in support]
    with localcontext() as ctx:
        ctx.prec = 80
        active = [i for i in support if profile.classes[i] is not None]
        total = sum(Decimal(str(weights[i])) for i in active)
        initial = {i:Decimal(str(weights[i]))/total for i in active}
        mass = initial.copy()
        bits = {i:profile.bits(key,context,profile.classes[i]) for i in active}
        for layer in range(profile.config.layers):
            updated = {i:Decimal(0) for i in active}
            for left in active:
                for right in active:
                    winner = right if bits[right][layer] > bits[left][layer] else left
                    updated[winner] += mass[left]*mass[right]
            mass = updated
        for token in active:
            expected = float(mass[token].ln()-initial[token].ln())
            actual = audit.independent_ratio(profile,key,context,support,probabilities,token)
            assert actual == pytest.approx(expected,abs=1e-10)


@pytest.mark.parametrize("ratio", [-1e6,-1000.,0.,1000.,1e6])
def test_half_mixture_remains_finite_without_probability_floor(audit,ratio):
    value = audit.half_factor(ratio)
    assert math.isfinite(value)
    if ratio < -100:
        assert value == -math.log(2.)
    elif ratio > 100:
        assert value == ratio-math.log(2.)
    else:
        assert value == 0.
