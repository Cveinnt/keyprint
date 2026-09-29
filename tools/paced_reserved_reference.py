"""Independent Fraction oracle for the reserved-margin numerical policy.

Unlike the prior output reference's multiplicative alpha backoff, reserve an
absolute base-relative margin inside every probability/movement bound BEFORE
labels. This is a separately identified numerical policy, not a silent change.
"""
from fractions import Fraction as F

from paced_budget_float_reference import normalized, certify, ROUNDING_MARGIN, MIN_BASE_PROBABILITY
from predictable_budget_oracle import validate


def strength(base, current):
    p,q=normalized(base),normalized(current)
    validate(p,q,F(2))
    if any(0<b<MIN_BASE_PROBABILITY for b in p):
        raise ValueError('Base probability below admitted research range')
    limits=[F(1)]; freeze=False
    for b,v in zip(p,q):
        if 0<v<1:
            lower=v-b/2-b*ROUNDING_MARGIN
            upper=2*b-v-b*ROUNDING_MARGIN
            if min(lower,upper)<=0: freeze=True
            else:
                radius=v*(1-v)
                limits.extend((lower/radius,upper/radius,(b/8-b*ROUNDING_MARGIN)/radius))
    return p,q,F(0) if freeze else min(limits),freeze


def step(base,current,bits):
    base,current=tuple(base),tuple(current)
    p,q,alpha,freeze=strength(base,current)
    bits=tuple(bits)
    if len(bits)!=len(p) or any(type(g) is not int or g not in (0,1) for g in bits):
        raise ValueError('Require binary labels matching weights')
    if alpha==0 or len({g for g,v in zip(bits,q) if v>0})<2:
        output=current
    else:
        mean=sum(v*g for v,g in zip(q,bits))
        output=tuple(float(v*(1+alpha*(g-mean))) for v,g in zip(q,bits))
    certify(base,current,output)
    return output,{'strength':alpha,'near_boundary_freeze':freeze}
