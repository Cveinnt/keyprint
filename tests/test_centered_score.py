from fractions import Fraction as F
from itertools import product
import pytest

from centered_score import tilt, expected_score_lift
from keyprint._engine.research.keyprint_exact_categorical_v2 import IntegerDistribution as D


def probabilities(d):
    return tuple(F(w,d.total) for w in d.weights)


@pytest.mark.parametrize('weights', [(1,1,1),(98,1,1),(1,10**40,1)])
def test_all_three_layer_patterns(weights):
    base=D(weights,sum(weights));p=probabilities(base)
    mean=[F(0)]*3
    for bits in product((0,1),repeat=9):
        scores=[sum(bits[i:i+3]) for i in range(0,9,3)]
        q=tilt(base,scores,3);qc=tilt(base,[3-s for s in scores],3)
        ps=probabilities(q);pc=probabilities(qc)
        for i in range(3):
            assert p[i]/2 <= ps[i] <= 3*p[i]/2
            assert (ps[i]+pc[i])/2==p[i]
            mean[i]+=ps[i]/512
        assert expected_score_lift(base,q,scores,3)>=-1e-16
    assert tuple(mean)==p


def test_excluded_mass_and_grouped_ratio():
    base=D((3,6,7,19),35);scores=[1,1,3,None]
    q=tilt(base,scores,3);p=probabilities(base);r=probabilities(q)
    assert r[-1]==p[-1] and r[0]/r[1]==p[0]/p[1]
    qc=tilt(base,[2,2,0,None],3)
    assert all((a+b)/2==v for a,b,v in zip(r,probabilities(qc),p))


@pytest.mark.parametrize('scores', [[None,None],[2,2],[0,None]])
def test_identity(scores):
    base=D((3,7),10)
    assert tilt(base,scores,3)==base


def test_exact_known_distribution_and_score():
    base=D((1,1),2);q=tilt(base,[0,30],30)
    assert probabilities(q)==(F(1,4),F(3,4))
    assert expected_score_lift(base,q,[0,30],30)==.25


@pytest.mark.parametrize('weights,total,scores,layers', [
    ((1,1),3,[0,1],1),((0,1),1,[0,1],1),((1,1),2,[0],1),
    ((1,1),2,[False,1],1),((1,1),2,[0,2],1),((1,1),2,[0,1],0)])
def test_invalid(weights,total,scores,layers):
    with pytest.raises(ValueError): tilt(D(weights,total),scores,layers)


@pytest.mark.parametrize('weights,scores', [
    ((1,3,7,10**60),[0,12,None,30]),
    ((99,1,1,1),[30,0,15,None]),
    ((1,1,1,1),[10,12,18,20]),
    ((1,1,2,7),[3,3,None,None])])
def test_independent_fraction_reference(weights,scores):
    base=D(weights,sum(weights));p=probabilities(base)
    active=[i for i,s in enumerate(scores) if s is not None]
    mass=sum(p[i] for i in active)
    mean=sum(p[i]*scores[i] for i in active)/mass
    radius=max(abs(scores[i]-mean) for i in active)
    expected=tuple(v if s is None or not radius else v*(1+(s-mean)/(2*radius))
                   for v,s in zip(p,scores))
    assert probabilities(tilt(base,scores,30))==expected
