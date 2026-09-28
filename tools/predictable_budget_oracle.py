"""Exact-rational research oracle for predictably bounded tournament updates.

Not an SDK implementation, detector, semantic guarantee or accepted candidate.
Choose each step size BEFORE inspecting that layer's bits. Never clip a completed
key-dependent proposal: clipping need not preserve its expectation.
"""
from fractions import Fraction


def validate(base, current, ratio):
    if (type(ratio) is not Fraction or ratio <= 1 or not base or len(base) != len(current)
            or any(type(x) is not Fraction or x < 0 for x in (*base, *current))
            or sum(base) != 1 or sum(current) != 1
            or any(not b / ratio <= q <= b * ratio for b, q in zip(base, current))):
        raise ValueError("Require exact normalized distributions within fixed ratio bounds")


def predictable_strength(base, current, ratio=Fraction(2)):
    """Largest common step safe for every possible upcoming binary label vector.

For q_i in (0,1), every binary assignment changes q_i by at most
alpha*q_i*(1-q_i), in either direction. The bound is conservative for grouped
labels. This function has no access to the upcoming bits.
"""
    validate(base, current, ratio)
    limits = [Fraction(1)]
    for b, q in zip(base, current):
        if 0 < q < 1:
            radius = q * (1 - q)
            limits.extend(((b * ratio - q) / radius, (q - b / ratio) / radius))
    return min(limits)


def step(base, current, bits, ratio=Fraction(2)):
    alpha = predictable_strength(base, current, ratio)
    if len(bits) != len(base) or any(type(g) is not int or g not in (0, 1) for g in bits):
        raise ValueError("Require binary vector matching distribution")
    mean = sum(q * g for q, g in zip(current, bits))
    output = tuple(q * (1 + alpha * (g - mean)) for q, g in zip(current, bits))
    validate(base, output, ratio)
    return output, alpha


def transform(base, layers, ratio=Fraction(2)):
    current, strengths = tuple(base), []
    for bits in layers:
        current, alpha = step(tuple(base), current, tuple(bits), ratio)
        strengths.append(alpha)
    return current, tuple(strengths)
