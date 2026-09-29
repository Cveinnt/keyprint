from fractions import Fraction as F
from itertools import product
import pytest
from balanced_score import allocate
from centered_score import tilt
from keyprint._engine.research.keyprint_exact_categorical_v2 import IntegerDistribution as D


def probs(d):return tuple(F(w,d.total) for w in d.weights)
def objective(d,scores):return sum(p*s for p,s in zip(probs(d),scores) if s is not None)


@pytest.mark.parametrize('weights',[(1,1,1),(98,1,1),(1,10**40,1)])
def test_complements_bounds_and_linear_domination(weights):
    base=D(weights,sum(weights));p=probs(base);mean=[F(0)]*3
    for bits in product((0,1),repeat=9):
        scores=[sum(bits[i:i+3]) for i in range(0,9,3)]
        q=allocate(base,scores,3);qc=allocate(base,[3-s for s in scores],3)
        for i,(x,y) in enumerate(zip(probs(q),probs(qc))):
            assert p[i]/2<=x<=3*p[i]/2 and (x+y)/2==p[i]
            mean[i]+=x/512
        assert objective(q,scores)>=objective(tilt(base,scores,3),scores)
    assert tuple(mean)==p


def test_exact_optimum_against_all_box_simplex_vertices():
    base=D((2,3,5),10);p=probs(base)
    for scores in product(range(4),repeat=3):
        q=allocate(base,scores,3)
        for free in range(3):
            fixed=[i for i in range(3) if i!=free]
            for endpoints in product((F(1,2),F(3,2)),repeat=2):
                vertex=[F(0)]*3
                for i,factor in zip(fixed,endpoints):vertex[i]=p[i]*factor
                vertex[free]=1-sum(vertex)
                if p[free]/2<=vertex[free]<=3*p[free]/2:
                    assert objective(q,scores)>=sum(v*s for v,s in zip(vertex,scores))


def test_ties_and_excluded_mass():
    base=D((3,6,7,19),35);scores=[1,1,3,None]
    q=allocate(base,scores,3);r=probs(q)
    assert r[-1]==F(19,35) and r[0]/r[1]==F(1,2)
    qc=allocate(base,[2,2,0,None],3)
    assert all((a+b)/2==p for a,b,p in zip(r,probs(qc),probs(base)))


@pytest.mark.parametrize('scores',[[None,None],[2,2],[0,None]])
def test_identity(scores):
    base=D((3,7),10)
    assert probs(allocate(base,scores,3))==probs(base)
