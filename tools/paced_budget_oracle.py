"""Exact research policy: spend at most one eighth of base mass per layer.

The original half-to-double bounds still apply. Strength depends on previous
states, never the upcoming labels. No SDK path or semantic guarantee.
"""
from fractions import Fraction as F

from predictable_budget_oracle import predictable_strength, validate

PER_LAYER_BASE_FRACTION = F(1, 8)


def paced_strength(base, current):
    safe = predictable_strength(base, current)
    caps = [safe]
    for b, q in zip(base, current):
        if 0 < q < 1:
            caps.append(PER_LAYER_BASE_FRACTION * b / (q * (1-q)))
    return min(caps)


def step(base, current, bits):
    alpha = paced_strength(base, current)
    if len(bits) != len(base) or any(type(g) is not int or g not in (0, 1) for g in bits):
        raise ValueError('Require binary vector matching distribution')
    mean = sum(q*g for q, g in zip(current, bits))
    output = tuple(q * (1 + alpha*(g-mean)) for q, g in zip(current, bits))
    validate(base, output, F(2))
    if any(abs(out-q) > PER_LAYER_BASE_FRACTION*b for b, q, out in zip(base, current, output)):
        raise AssertionError('Per-layer base-mass budget exceeded')
    return output, alpha


def transform(base, layers):
    base = tuple(base)
    current, strengths = base, []
    for bits in layers:
        current, alpha = step(base, current, tuple(bits))
        strengths.append(alpha)
    return current, tuple(strengths)
