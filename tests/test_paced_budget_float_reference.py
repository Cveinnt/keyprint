from fractions import Fraction as F
from itertools import product
import math
from pathlib import Path
import random
import sys

import pytest

sys.path.insert(0,str(Path(__file__).parents[1]/'tools'))
from paced_budget_float_reference import normalized, prepare, certify, step
from paced_budget_oracle import transform
from keyprint._engine.research.keyprint_exact_categorical_v2 import integer_distribution


def test_normalization_matches_exact_categorical_sampler_including_tiny_weights():
    weights=(1.,math.exp(-300),math.exp(-600),0.)
    actual=integer_distribution(weights)
    assert normalized(weights)==tuple(F(i,actual.total) for i in actual.weights)


@pytest.mark.parametrize('base', [(0.98,0.01,0.01),(0.5,0.25,0.25),(0.,0.25,0.75)])
def test_all_three_layer_paths_stay_near_exact_oracle_without_bound_exceptions(base):
    p=normalized(base); labels=tuple(product((0,1),repeat=3))
    average=[F(0)]*3
    for layers in product(labels,repeat=3):
        q=base
        for bits in layers:
            out,receipt=step(base,q,bits)
            assert receipt['maximum_base_relative_step']<=F(1,8)
            q=out
        expected,_=transform(p,layers)
        actual=normalized(q)
        assert max(abs(x-y)/b for x,y,b in zip(actual,expected,p) if b)<F(1,10**10)
        average=[x+y for x,y in zip(average,actual)]
    # Rounding has measured error, not an assertion of exact unbiasedness.
    assert max(abs(v/F(512)-b) for v,b in zip(average,p))<F(1,10**14)


@pytest.mark.parametrize('kind',['uniform','zipf','wide_tail','grouped'])
def test_thirty_layers_on_one_hundred_weight_distributions(kind):
    base=tuple(1. if kind=='uniform' else float(1+2*(i%2)) if kind=='grouped' else
        1./(i+1) if kind=='zipf' else math.exp(-600*i/99) for i in range(100))
    q=base; rng=random.Random(20260929)
    for _ in range(30):
        labels=[rng.randrange(2) for _ in range(50 if kind=='grouped' else 100)]
        bits=[labels[i//2] for i in range(100)] if kind=='grouped' else labels
        out,_=step(base,q,bits)
        certify(base,q,out)
        q=out
    if kind=='grouped': assert all(math.isclose(q[i+1]/q[i],3.,rel_tol=1e-14) for i in range(0,100,2))


def test_near_bound_freeze_is_before_labels_and_preserves_exact_weights():
    base=(0.5,0.5); current=(0.75,0.25)
    assert prepare(base,current)[2]==0
    for bits in product((0,1),repeat=2):
        out,r=step(base,current,bits)
        assert out==current and r['near_boundary_freeze']


def test_identical_labels_preserve_original_weights_without_rounding_again():
    base=(2.,1.,0.)
    for bits in ((1,1,0),(0,0,1)):
        out,_=step(base,base,bits)
        assert out==base


def test_source_weight_rescaling_does_not_change_categorical_distribution():
    a,_=step((.75,.25),(.75,.25),(1,0))
    b,_=step((3.,1.),(3.,1.),(1,0))
    assert normalized(a)==normalized(b)


@pytest.mark.parametrize('base',[(0.,0.),(float('nan'),1.),(float('inf'),1.),
    (-1.,2.),(True,1.),(1.,math.ulp(0.))])
def test_invalid_or_unadmitted_inputs_fail_before_output(base):
    with pytest.raises(ValueError): step(base,base,(0,1))


@pytest.mark.parametrize('current',[(.9,.1),(.5,0.),(.5,.25,.25)])
def test_changed_support_shape_or_escaped_bounds_rejected(current):
    with pytest.raises(ValueError): step((.5,.5),current,(0,1))


def test_certificate_rejects_renormalization_that_breaks_exact_bounds():
    # The raw second weight equals base/2, but normalization makes it smaller.
    with pytest.raises(ArithmeticError): certify((.5,.5),(.75,.25),(math.nextafter(.75,1.),.25))


@pytest.mark.parametrize('bits',[(True,0),(2,0),(1,)])
def test_nonbinary_or_wrong_size_labels_rejected(bits):
    with pytest.raises(ValueError): step((.5,.5),(.5,.5),bits)
