"""Numerical ideal-random-key tail for the fixed linear layer score.

For independent fair bits, X=sum(w*(2*b-1)) has characteristic function
prod(cos(w*t))**events. Invert it on a finite cyclic lattice. A Hoeffding bound
accounts for probability wrapping outside that lattice; an explicit numerical
margin is added. This is NOT a proof of floating-point error or fixed-key
deployment calibration. Independent enumeration and doubled-grid checks are
required. General DFT method: https://arxiv.org/abs/1702.01326
"""
from functools import lru_cache
import math
import numpy as np

WEIGHTS = tuple(290 - 9 * i for i in range(30))  # proportional to 10,...,1
ALIAS_TARGET = 1e-14
NUMERIC_MARGIN = 1e-9


@lru_cache(maxsize=4)
def characteristic(weights, size):
    frequencies = 2 * np.pi * np.arange(size // 2 + 1) / size
    values = np.ones(len(frequencies))
    for weight in weights:
        values *= np.cos(weight * frequencies)
    values.setflags(write=False)
    return values


def lattice(events, weights=WEIGHTS, *, double_grid=False):
    if type(events) is not int or not 1 <= events <= 2048:
        raise ValueError("Require 1-2048 eligible events")
    weights = tuple(weights)
    if not weights or any(type(w) is not int or w < 1 for w in weights):
        raise ValueError("Positive integer weights required")
    support = events * sum(weights)
    variance = events * sum(w*w for w in weights)
    half = min(support + 1, math.ceil(math.sqrt(2 * variance * math.log(2 / ALIAS_TARGET))))
    size = 1 << (2 * half).bit_length()
    if double_grid:
        size *= 2
    alias = 0. if size // 2 > support else 2 * math.exp(-(size / 2)**2 / (2 * variance))
    probability = np.fft.irfft(characteristic(weights, size)**events, n=size)
    mass_error = abs(float(probability.sum()) - 1)
    negative_mass = float(-probability[probability < 0].sum())
    if not np.isfinite(probability).all() or mass_error + negative_mass > NUMERIC_MARGIN:
        raise ArithmeticError("FFT mass check exceeds numerical allowance")
    # Clip negative roundoff, never renormalize it into a smaller upper tail.
    tail = np.cumsum(np.maximum(probability[:size // 2], 0)[::-1])[::-1]
    return tail, {"lattice_size": size, "alias_bound": alias,
                  "numeric_margin": NUMERIC_MARGIN, "mass_error": mass_error,
                  "negative_mass": negative_mass, "support": support}


@lru_cache(maxsize=8)
def cached_lattice(events):
    return lattice(events)


def tail_from_sum(centered_sum, events, *, double_grid=False):
    if type(centered_sum) is not int or type(events) is not int or not 1 <= events <= 2048:
        raise ValueError("Integer centered sum and 1-2048 events required")
    if abs(centered_sum) > events * sum(WEIGHTS):
        raise ValueError("Centered sum exceeds support")
    if centered_sum <= 0:
        return 1., {"scope": "nonpositive statistic conservatively returns one"}
    tails, audit = lattice(events, double_grid=True) if double_grid else cached_lattice(events)
    base = float(tails[centered_sum]) if centered_sum < len(tails) else 0.
    return min(1., base + audit["alias_bound"] + NUMERIC_MARGIN), audit


def reference_tail(bits, *, double_grid=False):
    bits = np.asarray(bits)
    if bits.ndim != 2 or bits.shape[1] != 30 or not len(bits) or not np.isin(bits, [0, 1]).all():
        raise ValueError("Nonempty binary matrix with 30 layers required")
    centered = int(((2 * bits.astype(np.int64) - 1) * np.array(WEIGHTS)).sum())
    value, audit = tail_from_sum(centered, len(bits), double_grid=double_grid)
    return {"reference_tail": value, "centered_sum": centered, "events": len(bits), **audit}


def two_key_tail(values):
    if len(values) != 2 or any(not math.isfinite(v) or not 0 <= v <= 1 for v in values):
        raise ValueError("Two finite reference tails required")
    return min(1., 2 * min(values))
