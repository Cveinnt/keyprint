"""Exact half-mass score allocation; research only, not a released profile.

Maximizes expected supplied score within [p/2,3p/2], fixed excluded probabilities
and normalization. Tied scores share mass proportionally. Complemented scores
reverse the half-mass allocation, so the paired distributions average to p.
"""
import math
from functools import reduce
from centered_score import tilt
from keyprint._engine.research.keyprint_exact_categorical_v2 import IntegerDistribution

POLICY='exact-balanced-layer-score-half-radius-v1'


def allocate(base,scores,layers):
    # Reuse strict input validation, not the linear proposal as a fallback.
    tilt(base,scores,layers)
    masses={}
    for b,s in zip(base.weights,scores):
        if s is not None:masses[s]=masses.get(s,0)+b
    mass=sum(masses.values());below=0;factors={}
    for s,g in sorted(masses.items()):
        lower_twice=min(2*g,max(0,mass-2*below))
        n,d=3*g-lower_twice,2*g
        divisor=math.gcd(n,d);factors[s]=(n//divisor,d//divisor)
        below+=g
    common=math.lcm(*(d for n,d in factors.values()))
    weights=tuple(b*common if s is None else b*factors[s][0]*(common//factors[s][1])
                  for b,s in zip(base.weights,scores))
    total=base.total*common;divisor=reduce(math.gcd,weights)
    result=IntegerDistribution(tuple(w//divisor for w in weights),total//divisor)
    if sum(result.weights)!=result.total:raise ArithmeticError('Total probability changed')
    for b,q,s in zip(base.weights,result.weights,scores):
        if not b*result.total<=2*q*base.total<=3*b*result.total:
            raise ArithmeticError('Half-to-one-and-a-half bound violated')
        if s is None and q*base.total!=b*result.total:raise ArithmeticError('Excluded probability changed')
    return result
