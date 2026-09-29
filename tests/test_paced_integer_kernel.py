from fractions import Fraction as F
from itertools import product
import math
from pathlib import Path
import random
import sys

import pytest

sys.path.insert(0,str(Path(__file__).parents[1]/'tools'))
from paced_integer_kernel import PacedKernel,distribution
from paced_reserved_reference import strength,step as reference
from paced_budget_float_reference import normalized,certify


@pytest.mark.parametrize('base',[(.98,.01,.01),(.5,.25,.25),(0.,.25,.75)])
def test_all_three_layer_paths_match_independent_reference(base):
    k=PacedKernel(base); p=normalized(base)
    for layers in product(tuple(product((0,1),repeat=3)),repeat=3):
        q=base
        for bits in layers:
            _,_,exact,frozen=strength(base,q)
            alpha,freeze=k._strength(distribution(q))
            assert freeze==frozen and F(alpha)<=exact
            assert exact-F(alpha)<=F(math.ulp(alpha))
            expected,_=reference(base,q,bits)
            out,_=k.step(q,bits)
            actual,want=normalized(out),normalized(expected)
            assert max(abs(a-b)/p_i for a,b,p_i in zip(actual,want,p) if p_i)<F(1,10**13)
            certify(base,q,out)
            q=out


@pytest.mark.parametrize('kind',['uniform','zipf','wide-tail','grouped'])
def test_thirty_layer_full_trajectories_match_reference(kind):
    base=tuple(1. if kind=='uniform' else 1./(i+1) if kind=='zipf' else
        math.exp(-600*i/99) if kind=='wide-tail' else float(1+2*(i%2)) for i in range(100))
    k=PacedKernel(base); q=slow=base; p=normalized(base); rng=random.Random(20260929)
    for _ in range(30):
        labels=[rng.randrange(2) for _ in range(50 if kind=='grouped' else 100)]
        bits=[labels[i//2] for i in range(100)] if kind=='grouped' else labels
        q,_=k.step(q,bits); slow,_=reference(base,slow,bits)
        assert max(abs(a-b)/p_i for a,b,p_i in zip(normalized(q),normalized(slow),p))<F(1,10**10)


def test_near_boundary_freeze_and_identity_preserve_weights():
    k=PacedKernel((.5,.5))
    for bits in product((0,1),repeat=2):
        out,r=k.step((.75,.25),bits)
        assert out==(.75,.25) and r['near_boundary_freeze']
    assert PacedKernel((2.,1.,0.)).step((2.,1.,0.),(1,1,0))[0]==(2.,1.,0.)


def test_integer_certificate_rejects_normalization_crossing_bound():
    k=PacedKernel((.5,.5))
    with pytest.raises(ArithmeticError):
        k._certify(distribution((.75,.25)),distribution((math.nextafter(.75,1.),.25)))


@pytest.mark.parametrize('weights',[(1.,math.ulp(0.)),(float('nan'),1.),(float('inf'),1.),
    (0.,0.),(-1.,2.),(True,1.)])
def test_invalid_base_rejected(weights):
    with pytest.raises((ValueError,TypeError)): PacedKernel(weights)


@pytest.mark.parametrize('current',[(.9,.1),(.5,0.),(.5,.25,.25)])
def test_invalid_current_state_rejected(current):
    with pytest.raises(ValueError): PacedKernel((.5,.5)).step(current,(0,1))


@pytest.mark.parametrize('bits',[(True,0),(2,0),(1,)])
def test_invalid_labels_rejected(bits):
    with pytest.raises(ValueError): PacedKernel((.5,.5)).step((.5,.5),bits)
