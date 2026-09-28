from fractions import Fraction as F
from itertools import product
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "tools"))
from predictable_budget_oracle import predictable_strength, step, transform


@pytest.mark.parametrize('base', [(F(98,100), F(1,100), F(1,100)),
    (F(1,2), F(1,3), F(1,6)), (F(0), F(1,4), F(3,4))])
def test_every_three_layer_bit_path_preserves_bounds_and_exact_random_bit_mean(base):
    labels = list(product((0, 1), repeat=3))
    total = [F(0)] * 3
    for layers in product(labels, repeat=3):
        q, alphas = transform(base, layers)
        assert sum(q) == 1
        assert all(b / 2 <= v <= 2 * b for b, v in zip(base, q))
        assert all(0 <= a <= 1 for a in alphas)
        total = [a + b for a, b in zip(total, q)]
    assert tuple(x / len(labels)**3 for x in total) == base


def test_no_strength_decision_depends_on_upcoming_bits():
    base = (F(99,100), F(1,100))
    expected = predictable_strength(base, base)
    assert {step(base, base, bits)[1] for bits in product((0,1), repeat=2)} == {expected}


def test_thirty_identical_favorable_layers_cannot_amplify_rare_token_above_bound():
    base = (F(999,1000), F(1,1000))
    out, strengths = transform(base, [(0,1)] * 30)
    assert out[1] == F(2,1000)
    assert strengths[-1] == 0  # A hard bound can exhaust all remaining capacity.
    assert sum(out) == 1


def test_grouped_labels_preserve_within_group_ratios_and_random_bit_mean():
    base = (F(1,8), F(3,8), F(1,2))
    total = [F(0)] * 3
    for g, h in product((0,1), repeat=2):
        q, _ = step(base, base, (g,g,h))
        assert q[0] / q[1] == F(1,3)
        total = [a+b for a,b in zip(total,q)]
    assert tuple(x/4 for x in total) == base


def test_clipping_after_seeing_bits_is_not_an_unbiased_substitute():
    base = (F(3,5), F(2,5))
    total = [F(0), F(0)]
    for bits in product((0,1), repeat=2):
        mean = sum(p*g for p,g in zip(base,bits))
        proposal = [p*(1+g-mean) for p,g in zip(base,bits)]
        clipped = [max(p/2,min(2*p,q)) for p,q in zip(base,proposal)]
        normalized = [q/sum(clipped) for q in clipped]
        total = [a+b for a,b in zip(total,normalized)]
    assert tuple(x/4 for x in total) != base


@pytest.mark.parametrize('base,current,ratio', [((F(1),),(F(1),),F(1)),
    ((F(1,2),F(1,2)),(F(1),F(0)),F(2)), ((F(1),),(F(2),),F(2)),
    ((1.,),(1.,),F(2)), ((F(1),),(F(1),),2)])
def test_invalid_or_inexact_state_rejected(base,current,ratio):
    with pytest.raises(ValueError): predictable_strength(base,current,ratio)
