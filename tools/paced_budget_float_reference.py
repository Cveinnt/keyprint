"""Certified binary64-output research reference, not a fast SDK implementation.

Exact rational arithmetic chooses strength and forms the proposal. Only output
weights are rounded to binary64. Bounds are checked on exact normalized ratios
of represented weights, matching the categorical sampler's interpretation.
No post-label clipping, retry, flooring or fallback to another distribution.
"""
from fractions import Fraction as F
import math

from paced_budget_oracle import paced_strength

ROUNDING_MARGIN = F(1, 2**40)
MIN_BASE_PROBABILITY = F(1, 2**900)


def normalized(values):
    values = tuple(values)
    if (not 1 <= len(values) <= 4096 or any(type(x) is not float for x in values)
            or any(not math.isfinite(x) or x < 0 for x in values)):
        raise ValueError('Require 1 to 4096 finite nonnegative binary64 weights')
    exact = tuple(F(x) for x in values)
    total = sum(exact)
    if total <= 0: raise ValueError('Positive mass required')
    return tuple(x/total for x in exact)


def prepare(base, current):
    """No access to upcoming bits. A near-bound state freezes predictably."""
    p, q = normalized(base), normalized(current)
    if len(p) != len(q) or any((b>0)!=(v>0) for b,v in zip(p,q)):
        raise ValueError('Shape or support changed')
    if any(0 < b < MIN_BASE_PROBABILITY for b in p):
        raise ValueError('Base probability below admitted research range')
    alpha = paced_strength(p,q)  # Also validates exact total half-to-double bounds.
    near = any(b>0 and min(v-b/2,2*b-v) <= b*ROUNDING_MARGIN for b,v in zip(p,q))
    if near: alpha=F(0)
    else: alpha *= 1-ROUNDING_MARGIN
    return p,q,alpha,near


def certify(base, before, after):
    """Certify the distribution actually sampled after binary64 rounding."""
    p,q,r = normalized(base), normalized(before), normalized(after)
    if len(p)!=len(q) or len(q)!=len(r): raise ArithmeticError('Shape changed')
    if any((b>0)!=(v>0) for b,v in zip(p,r)):
        raise ArithmeticError('Positive support lost or introduced')
    if any(not b/2 <= out <= 2*b or abs(out-old)>b/8 for b,old,out in zip(p,q,r)):
        raise ArithmeticError('Rounded categorical probabilities violate budget')
    return r


def step(base, current, bits):
    base, current = tuple(base), tuple(current)
    p,q,alpha,near = prepare(base,current)
    bits=tuple(bits)
    if len(bits)!=len(p) or any(type(g) is not int or g not in (0,1) for g in bits):
        raise ValueError('Require binary labels matching weights')
    positive_labels={g for g,v in zip(bits,q) if v>0}
    if alpha==0 or len(positive_labels)<2:
        proposal=q
        output=current  # Preserve represented weights, including exact identity.
    else:
        mean=sum(v*g for v,g in zip(q,bits))
        proposal=tuple(v*(1+alpha*(g-mean)) for v,g in zip(q,bits))
        output=tuple(float(v) for v in proposal)
    actual=certify(base,current,output)
    return output, {'strength':alpha,'near_boundary_freeze':near,
        'rounding_total_variation':sum(abs(a-b) for a,b in zip(actual,proposal))/2,
        'maximum_base_relative_step':max((abs(a-b)/p_i for a,b,p_i in zip(actual,q,p) if p_i),default=F(0))}
