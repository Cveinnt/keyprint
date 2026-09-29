"""Exact research-only centered-score tilt, not a released sampling profile.

Scores are sums of the same number of binary layers. Global complementation
negates centered scores while preserving their scale; complementary outputs
average exactly to base. This ideal-label argument is not fixed-key semantics
or finite-PRF unbiasedness. No per-layer p/8 guarantee is claimed.
"""
from functools import reduce
import math

from keyprint._engine.research.keyprint_exact_categorical_v2 import IntegerDistribution

POLICY = 'exact-centered-layer-score-half-radius-v1'


def tilt(base, scores, layers):
    """None scores exclude tokens, preserving their global probabilities exactly."""
    if (type(layers) is not int or layers <= 0 or len(scores) != len(base.weights)
            or type(base.total) is not int or base.total <= 0
            or not base.weights or sum(base.weights) != base.total
            or any(type(b) is not int or b <= 0 for b in base.weights)
            or any(s is not None and (type(s) is not int or not 0 <= s <= layers)
                   for s in scores)):
        raise ValueError('Positive integer distribution and aligned layer scores required')
    active = [(b,s) for b,s in zip(base.weights,scores) if s is not None]
    mass = sum(b for b,s in active)
    weighted = sum(b*s for b,s in active)
    deviations = [0 if s is None else mass*s-weighted for s in scores]
    radius = max(map(abs,deviations))
    if not radius:
        return base
    weights = tuple(b*(2*radius+d) for b,d in zip(base.weights,deviations))
    total = base.total*2*radius
    divisor = reduce(math.gcd,weights)
    result = IntegerDistribution(tuple(w//divisor for w in weights),total//divisor)
    if sum(result.weights) != result.total:
        raise ArithmeticError('Total probability changed')
    for b,q,s in zip(base.weights,result.weights,scores):
        if not b*result.total <= 2*q*base.total <= 3*b*result.total:
            raise ArithmeticError('Half-to-one-and-a-half ratio bound violated')
        if s is None and q*base.total != b*result.total:
            raise ArithmeticError('Excluded probability changed')
    return result


def expected_score_lift(base, marked, scores, layers):
    """Same-label expected score difference; excludes None scores in both arms."""
    return math.fsum((q*base.total-b*marked.total)*s /
                     (base.total*marked.total*layers)
                     for b,q,s in zip(base.weights,marked.weights,scores,strict=True)
                     if s is not None)
