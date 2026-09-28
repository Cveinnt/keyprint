from pathlib import Path
import sys
import math
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "tools"))
from diagnose_source_sampling import distribution_metrics, aggregate


def test_ordinary_same_distribution_and_eos_are_unchanged():
    p = np.array([.5, .25, 0, .25], dtype=np.float64)
    r = distribution_metrics(p, p.copy(), 3, {3})
    assert r['base_entropy_bits'] == r['effective_entropy_bits'] == 1.5
    assert r['total_variation'] == r['kl_effective_to_base_nats'] == 0
    assert r['selected_base_rank'] == 2
    assert r['eos_base_mass'] == r['eos_effective_mass'] == .25
    assert r['selected_is_eos']


def test_concentration_and_low_probability_amplification_are_measured():
    p = np.array([.999, .001], dtype=np.float64)
    q = np.array([0., 1.], dtype=np.float64)
    r = distribution_metrics(p, q, 1, {0})
    assert r['selected_base_probability'] == .001 and r['selected_effective_probability'] == 1
    assert r['kl_effective_to_base_nats'] == pytest.approx(math.log(1000))
    assert r['total_variation'] == pytest.approx(.999)
    assert r['effective_entropy_bits'] == 0 and r['eos_effective_mass'] == 0
    assert aggregate([r])['selected_base_below_1pct'] == 1


def test_telemetry_normalization_does_not_modify_sampling_inputs():
    p, q = np.array([2., 2.]), np.array([0., 8.])
    r = distribution_metrics(p, q, 1, set())
    np.testing.assert_array_equal(p, [2., 2.]); np.testing.assert_array_equal(q, [0., 8.])
    assert r['total_variation'] == .5
    assert aggregate([r])['newly_above_99pct'] == 1
    assert aggregate([]) == {'steps': 0}


@pytest.mark.parametrize('p,q,token', [([0.,1.],[1.,0.],0), ([1.,0.],[0.,1.],1),
    ([1.,0.],[1.,0.],1), ([0.,0.],[0.,0.],0), ([math.nan,1.],[.5,.5],0),
    ([-1.,2.],[0.,1.],1), ([1.,0.],[1.,0.],3)])
def test_changed_support_invalid_weights_or_impossible_token_rejected(p,q,token):
    with pytest.raises(ValueError): distribution_metrics(np.array(p),np.array(q),token,set())
