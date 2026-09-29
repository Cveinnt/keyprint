from fractions import Fraction as F
from itertools import product
import math
import random

import pytest

from complement_paced import exact_step, exact_strength, reference_step, ComplementPacedKernel
from paced_budget_oracle import step as previous
from paced_budget_float_reference import normalized


@pytest.mark.parametrize('base',[(F(1,2),F(1,3),F(1,6)),(F(98,100),F(1,100),F(1,100)),(F(0),F(1,4),F(3,4))])
def test_all_three_layer_paths_exact_marginal_and_bounds(base):
    labels=tuple(product((0,1),repeat=3));total=[F(0)]*3
    for layers in product(labels,repeat=3):
        q=base
        for bits in layers:
            out,a=exact_step(base,q,bits)
            opposite,b=exact_step(base,q,tuple(1-g for g in bits))
            assert a==b and tuple((x+y)/2 for x,y in zip(out,opposite))==q
            assert all(p/2<=v<=2*p and abs(v-old)<=p/8 for p,v,old in zip(base,out,q))
            q=out
        total=[a+b for a,b in zip(total,q)]
    assert tuple(v/512 for v in total)==base


def test_grouped_labels_preserve_ratios():
    base=(F(1,8),F(3,8),F(1,2))
    for g,h in product((0,1),repeat=2):
        q,_=exact_step(base,base,(g,g,h));assert q[0]/q[1]==F(1,3)


def test_actual_split_can_use_more_strength_without_bigger_budget():
    base=(F(1,4),)*4;bits=(0,0,1,1)
    q,a=exact_step(base,base,bits);old,b=previous(base,base,bits)
    assert a>b and q==(F(7,32),F(7,32),F(9,32),F(9,32))


@pytest.mark.parametrize('n,kind',[(3,'uniform'),(100,'uniform'),(100,'wide'),(100,'peaked')])
def test_thirty_layer_numeric_reference_and_complement_strength(n,kind):
    base=tuple(1. if kind=='uniform' else math.exp(-600*i/(n-1)) if kind=='wide' else
               .99 if i==0 else .01/(n-1) for i in range(n))
    kernel=ComplementPacedKernel(base);fast=slow=base;rng=random.Random(17)
    for _ in range(30):
        bits=[rng.randrange(2) for _ in range(n)]
        opposite=[1-g for g in bits]
        _,a=kernel.step(fast,bits);_,b=kernel.step(fast,opposite)
        assert a['strength']==b['strength']
        fast,_=kernel.step(fast,bits);slow,_=reference_step(base,slow,bits)
        assert max(abs(x-y)/p for x,y,p in zip(normalized(fast),normalized(slow),normalized(base)))<F(1,10**10)


def test_rounded_pair_error_is_small_not_claimed_zero():
    base=(.1,.2,.3,.4);kernel=ComplementPacedKernel(base);p=normalized(base)
    for bits in product((0,1),repeat=4):
        a,_=kernel.step(base,bits);b,_=kernel.step(base,tuple(1-g for g in bits))
        assert max(abs((x+y)/2-v)/v for x,y,v in zip(normalized(a),normalized(b),p))<F(1,10**14)


def test_rounded_two_layer_marginal_over_all_64_label_paths():
    base=(.5,.3,.2);kernel=ComplementPacedKernel(base);total=[F(0)]*3
    labels=tuple(product((0,1),repeat=3))
    for layers in product(labels,repeat=2):
        q=base
        for bits in layers:q,_=kernel.step(q,bits)
        total=[a+b for a,b in zip(total,normalized(q))]
    assert max(abs(t/64-p)/p for t,p in zip(total,normalized(base)))<F(1,10**14)


@pytest.mark.parametrize('seed',[0,1])
def test_extreme_tail_boundary_regression_matches_exact_proposal(seed):
    # Both frozen numerical fixtures exposed divergent multi-round trajectories.
    base=tuple(math.exp(-600*i/99) for i in range(100))
    kernel=ComplementPacedKernel(base);fast=slow=base;rng=random.Random(seed)
    for _ in range(30):
        bits=[rng.randrange(2) for _ in base]
        fast,_=kernel.step(fast,bits);slow,_=reference_step(base,slow,bits)
        assert fast==slow


def test_invalid_bits_margin_and_current_fail():
    p=(F(1,2),F(1,2))
    for bits in ((True,0),(0,2),(0,)):
        with pytest.raises(ValueError):exact_step(p,p,bits)
    with pytest.raises(ValueError):exact_strength(p,p,(0,1),margin=F(1,8))
    with pytest.raises(ValueError):ComplementPacedKernel((.5,.5)).step((.99,.01),(0,1))


def test_boundary_and_adversarial_paths_preserve_support():
    base=(F(1,2),F(1,3),F(1,6));q=base
    for _ in range(30):q,_=exact_step(base,q,(1,0,0))
    assert all(p/2<=x<=2*p for p,x in zip(base,q))
    at_boundary=(F(1,4),F(1,2),F(1,4))
    out,a=exact_step(base,at_boundary,(0,1,1))
    assert a==0 and out==at_boundary
