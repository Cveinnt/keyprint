"""Proposed future sampler. Not installed or bound to the frozen Keyprint profile.

Samples exact normalized ratios of represented finite binary64 weights, using
unbiased variable-length random bits supplied explicitly by the caller. This is
not an exact-real softmax implementation and does not fix underflow by flooring.
"""
from dataclasses import dataclass
from functools import reduce
import math
from typing import Callable, Sequence


class SupportLossError(ArithmeticError):
    """A finite post-filter logit lost positive probability before sampling."""


@dataclass(frozen=True)
class IntegerDistribution:
    weights: tuple[int, ...]
    total: int


@dataclass(frozen=True)
class BitDraw:
    bit_count: int
    value: int
    accepted: bool


@dataclass(frozen=True)
class Sample:
    token_index: int
    integer_point: int
    total_weight: int
    transcript: tuple[BitDraw, ...]


def _floats(values: Sequence[float]) -> tuple[float, ...]:
    result = tuple(values)
    if not result or any(type(value) is not float for value in result):
        raise TypeError("nonempty sequence of Python binary64 floats required")
    return result


def integer_distribution(weights: Sequence[float]) -> IntegerDistribution:
    """Clear binary denominators, then remove their common integer factor.

    For every i, result.weights[i] / result.total is exactly
    weights[i] / sum(weights) with the sum interpreted over rational numbers.
    No floating-point normalization, cumulative sum, or support floor occurs.
    """
    values = _floats(weights)
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("weights must be finite and nonnegative")
    ratios = tuple(value.as_integer_ratio() for value in values)
    denominator = max(d for _, d in ratios)  # Binary64 denominators are powers of 2.
    integers = tuple(n * (denominator // d) for n, d in ratios)
    divisor = reduce(math.gcd, integers)
    if divisor == 0:
        raise ValueError("at least one positive weight required")
    integers = tuple(value // divisor for value in integers)
    return IntegerDistribution(integers, sum(integers))


def supported_softmax(filtered_logits: Sequence[float]) -> tuple[float, ...]:
    """Fail before sampling if ordinary binary64 softmax loses finite support.

    Minus infinity is the only accepted exclusion marker. This guard rejects
    unsupported numerical inputs instead of claiming to recover their tails.
    """
    logits = _floats(filtered_logits)
    if any(math.isnan(value) or value == math.inf for value in logits):
        raise ValueError("logits must be finite or negative infinity")
    finite = tuple(math.isfinite(value) for value in logits)
    if not any(finite):
        raise ValueError("at least one finite post-filter logit required")
    maximum = max(logits)
    weights = tuple(math.exp(value - maximum) if included else 0.0
                    for value, included in zip(logits, finite))
    total = math.fsum(weights)
    probabilities = tuple(value / total for value in weights)
    lost = tuple(i for i, (included, value) in enumerate(zip(finite, probabilities))
                 if included != (value > 0))
    if lost:
        raise SupportLossError(f"binary64 softmax lost post-filter support at indices {lost}")
    return probabilities


def index_at_integer_point(distribution: IntegerDistribution, point: int) -> int:
    """Integer interval lookup; zero-weight tokens have empty intervals."""
    if type(point) is not int or not 0 <= point < distribution.total:
        raise ValueError("integer point outside distribution")
    for index, weight in enumerate(distribution.weights):
        if point < weight:
            return index
        point -= weight
    raise AssertionError("invalid integer distribution")


def sample_float_weights(weights: Sequence[float], random_bits: Callable[[int], int],
                         *, max_draws: int = 1024) -> Sample:
    """Sample represented weights without silently losing positive support.

    random_bits(k) must provide an independent uniform integer in [0, 2**k).
    Range/type validation cannot establish that a caller's source is unbiased.
    Rejection removes modulo bias. A draw cap raises without returning a token.
    The returned transcript enables deterministic replay and may expose caller
    randomness; this prototype does not automatically persist it.
    """
    distribution = integer_distribution(weights)
    if not callable(random_bits):
        raise TypeError("explicit callable random-bit source required")
    if type(max_draws) is not int or max_draws <= 0:
        raise ValueError("max_draws must be a positive integer")
    bits = (distribution.total - 1).bit_length()
    draws = []
    if bits == 0:
        point = 0
    else:
        for _ in range(max_draws):
            point = random_bits(bits)
            if type(point) is not int or not 0 <= point < (1 << bits):
                raise ValueError("random-bit source returned an invalid integer")
            accepted = point < distribution.total
            draws.append(BitDraw(bits, point, accepted))
            if accepted:
                break
        else:
            raise RuntimeError("random-bit rejection cap reached; no token sampled")
    return Sample(index_at_integer_point(distribution, point), point,
                  distribution.total, tuple(draws))


def sample_filtered_logits(filtered_logits: Sequence[float], random_bits: Callable[[int], int],
                           *, max_draws: int = 1024) -> Sample:
    """Guard finite support first, then sample; no model or commit side effects."""
    probabilities = supported_softmax(filtered_logits)
    return sample_float_weights(probabilities, random_bits, max_draws=max_draws)
