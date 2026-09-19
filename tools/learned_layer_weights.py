"""Development-only marginal layer weights with an ideal fair-bit null tail.

Fit positive log odds on marked training events under other keys. The reference
tail accounts for the resulting fixed integer weights; it does not make a
trained score deployment-calibrated or establish independent real-world bits.
"""
from functools import lru_cache
import math

import numpy as np

from layer_likelihood import binary_matrix
from weighted_null import lattice, NUMERIC_MARGIN

MAX_WEIGHT = 256
PRIOR_COUNT = .5


def fit_weights(bits):
    bits = binary_matrix(bits)
    probabilities = (bits.sum(axis=0) + PRIOR_COUNT) / (len(bits) + 2 * PRIOR_COUNT)
    # Nonpositive estimated evidence receives the minimum integer weight.
    # The .75 ceiling is a fixed regularizer, not an empirical guarantee.
    clipped = np.clip(probabilities, .5, .75)
    odds = np.log(clipped / (1 - clipped))
    if float(odds.max()) <= 0:
        raise ValueError("Training events contain no positive marginal signal")
    weights = np.maximum(1, np.rint(MAX_WEIGHT * odds / odds.max())).astype(int)
    return {"weights_integer": weights.tolist(), "marginal_probabilities": probabilities.tolist(),
            "fit_events": len(bits), "max_weight": MAX_WEIGHT, "prior_count": PRIOR_COUNT,
            "recipe": "Jeffreys-smoothed positive marginal log odds; clip probabilities to [.5,.75]; normalize maximum to 256; round to integers with floor 1",
            "scope": "Opened-data fitted weights, not an SDK detector or deployment qualification"}


def checked_weights(weights):
    weights = tuple(weights)
    if not weights or any(type(w) is not int or not 1 <= w <= MAX_WEIGHT for w in weights):
        raise ValueError("Integer weights in [1,256] required")
    return weights


@lru_cache(maxsize=8)
def cached_lattice(events, weights):
    return lattice(events, weights)


def reference_tail(bits, weights, *, double_grid=False):
    bits = binary_matrix(bits)
    weights = checked_weights(weights)
    if bits.shape[1] != len(weights) or not 1 <= len(bits) <= 2048:
        raise ValueError("Weights must match layers; 1-2048 eligible events required")
    centered = int(((2 * bits.astype(np.int64) - 1) * np.array(weights)).sum())
    result = {"centered_sum": centered, "events": len(bits)}
    if centered <= 0:
        return {**result, "reference_tail": 1., "scope": "Nonpositive statistic conservatively returns one"}
    # Common integer scale carries no information. Remove it before the FFT,
    # avoiding a needlessly sparse/larger grid and its extra rounding error.
    divisor = math.gcd(*weights)
    reduced = tuple(w // divisor for w in weights)
    cut = centered // divisor
    tails, audit = lattice(len(bits), reduced, double_grid=True) if double_grid else cached_lattice(len(bits), reduced)
    base = float(tails[cut]) if cut < len(tails) else 0.
    value = min(1., base + audit["alias_bound"] + NUMERIC_MARGIN)
    if not math.isfinite(value):
        raise ArithmeticError("Nonfinite reference tail")
    return {**result, "reference_tail": value, "weight_divisor": divisor,
            "lattice_centered_sum": cut, **audit}
