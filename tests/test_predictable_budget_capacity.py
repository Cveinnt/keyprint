from fractions import Fraction as F
from itertools import product
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]/'tools'))
from predictable_budget_capacity import analyze
from predictable_budget_oracle import transform


@pytest.mark.parametrize('base', [(F(1,2),F(1,2)), (F(99,100),F(1,100)),
                                  (F(1,2), F(1,3), F(1,6))])
def test_expected_final_token_score_matches_exhaustive_three_layer_paths(base):
    labels = tuple(product((0, 1), repeat=len(base)))
    total = F(0)
    paths = 0
    for layers in product(labels, repeat=3):
        final, _ = transform(base, layers)
        total += sum(final[i] * sum(bits[i] for bits in layers) for i in range(len(base)))/3
        paths += 1
    result = analyze(base, layers=3)
    assert result['expected_matching_bit_fraction'] == total/paths
    assert result['exact_final_marginal'] == base


def test_uniform_two_token_capacity_matches_closed_form_thirty_layers():
    result = analyze((F(1,2), F(1,2)))
    # First differing labels absorb at (.75,.25) or (.25,.75).
    # Probability no differing labels in L layers is 2^-L.
    assert result['expected_matching_bit_fraction'] == F(1,2)+F(1,4)*(1-F(1,2)**30)/30
    assert result['probability_strength_exhausted_after_last_layer'] == 1-F(1,2)**30
    assert result['expected_layers_with_positive_strength'] == 2*(1-F(1,2)**30)


def test_state_budget_exhaustion_fails_instead_of_dropping_low_mass_states():
    with pytest.raises(ValueError, match='State budget exceeded'):
        analyze((F(1,2), F(1,2)), max_states=1)


@pytest.mark.parametrize('layers', [0, 31, 1.5])
def test_unbounded_or_invalid_layer_requests_rejected(layers):
    with pytest.raises(ValueError): analyze((F(1),), layers=layers)
