from collections import defaultdict
from fractions import Fraction as F
from itertools import product
from pathlib import Path
import sys
import json

import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT/'tools'))
from paced_budget_oracle import paced_strength, step, transform
from predictable_budget_capacity import analyze, serializable


@pytest.mark.parametrize('base', [(F(98,100),F(1,100),F(1,100)),
    (F(1,2),F(1,3),F(1,6)), (F(0),F(1,4),F(3,4))])
def test_exhaustive_paths_preserve_marginal_bounds_and_layer_budget(base):
    labels = tuple(product((0,1), repeat=3))
    total = [F(0)] * 3
    score = F(0)
    for layers in product(labels, repeat=3):
        q = base
        for bits in layers:
            following, alpha = step(base, q, bits)
            assert sum(following) == 1 and 0 <= alpha <= 1
            assert all(b/2 <= out <= 2*b and abs(out-old) <= b/8
                for b, old, out in zip(base, q, following))
            q = following
        total = [a+b for a,b in zip(total,q)]
        score += sum(q[i] * sum(g[i] for g in layers) for i in range(3))/3
    assert tuple(v / 512 for v in total) == base
    analysis = analyze(base, layers=3, policy='paced-eighth')
    assert analysis['expected_matching_bit_fraction'] == score/512


def test_strength_cannot_depend_on_upcoming_labels():
    base = (F(99,100),F(1,100))
    assert {step(base, base, bits)[1] for bits in product((0,1),repeat=2)} == {paced_strength(base,base)}


def test_thirty_layer_uniform_result_matches_independent_integer_random_walk():
    # q=(1/2+k/16,1/2-k/16), absorbing at k=+-4. Each label layer:
    # stay with probability 1/2, otherwise move one lattice unit each direction.
    states = {0:F(1)}
    expected, active_sum = F(0), F(0)
    for _ in range(30):
        active = sum(m for k,m in states.items() if abs(k)<4)
        expected += F(1,2)+active/F(32)
        active_sum += active
        following = defaultdict(F)
        for k,m in states.items():
            if abs(k)==4: following[k] += m
            else:
                following[k] += m/2
                following[k-1] += m/4
                following[k+1] += m/4
        states = dict(following)
    result = analyze((F(1,2),F(1,2)),policy='paced-eighth')
    assert result['expected_matching_bit_fraction'] == expected/30
    assert result['expected_layers_with_positive_strength'] == active_sum
    assert result['probability_strength_exhausted_after_last_layer'] == states[-4]+states[4]


def test_grouped_labels_keep_within_group_ratios_and_exact_conditional_mean():
    base = (F(1,8),F(3,8),F(1,2))
    total = [F(0)]*3
    for g,h in product((0,1),repeat=2):
        q,_ = step(base,base,(g,g,h))
        assert q[0]/q[1] == F(1,3)
        total = [a+b for a,b in zip(total,q)]
    assert tuple(x/4 for x in total)==base


def test_thirty_adversarial_layers_still_cannot_bypass_total_bounds():
    base = (F(999,1000),F(1,1000))
    q, strengths = transform(base, [(0,1)]*30)
    assert q[1] == F(2,1000) and strengths[-1] == 0
    assert all(a>0 for a in strengths[:8])


def test_previous_maximum_policy_results_remain_byte_value_equivalent():
    old=json.loads((ROOT/'evidence/predictable-budget-capacity-2026-09-29/results.json').read_text())
    for row in old['rows']:
        base=tuple(F(x['exact']) for x in row['base'])
        assert serializable(analyze(base))==row


def test_unknown_policy_fails_instead_of_silently_using_default():
    with pytest.raises(ValueError, match='Unknown'):
        analyze((F(1),),policy='typo')
