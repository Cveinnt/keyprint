"""Sparse execution of the reference binary64 law for portable adapters.

Only excluded logits and zero weights are removed from arithmetic. Finite
logits keep their original order and use the reference math.exp/math.fsum
implementation. Sampling returns the original vocabulary index and preserves
the exact integer distribution and random-bit transcript. The frozen engine
is unchanged; adapters record this module's source identity separately.
"""
from dataclasses import replace
import hashlib
from pathlib import Path
from typing import Callable

import numpy as np

from ._engine.research.keyprint_exact_categorical_v2 import (
    Sample, sample_float_weights, supported_softmax,
)


def identity() -> dict[str, str]:
    return {"implementation": "sparse-reference-binary64-v1",
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "sample_index_scope": "original_vocabulary_token_id"}


def _vector(values: np.ndarray) -> None:
    if (not isinstance(values, np.ndarray) or values.dtype != np.float64
            or values.ndim != 1 or not 0 < values.size <= 151936):
        raise ValueError("expected a nonempty bounded float64 vocabulary vector")


def sparse_softmax(logits: np.ndarray) -> np.ndarray:
    """Compute reference probabilities without Python loops over exclusions."""
    _vector(logits)
    if np.isnan(logits).any() or np.isposinf(logits).any():
        raise ValueError("logits must be finite or negative infinity")
    support = np.flatnonzero(np.isfinite(logits))
    # Keep the reference's support-loss check, including underflow rejection.
    if not support.size:
        raise ValueError("at least one finite post-filter logit required")
    probabilities = supported_softmax(tuple(map(float, logits[support])))
    result = np.zeros(logits.size, dtype=np.float64)
    result[support] = probabilities
    return result


def sparse_sample(weights: np.ndarray, random_bits: Callable[[int], int],
                  *, max_draws: int = 1024) -> Sample:
    """Remove zero intervals, sample exactly, then restore the token index."""
    _vector(weights)
    if not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("weights must be finite and nonnegative")
    support = np.flatnonzero(weights > 0)
    if not support.size:
        raise ValueError("at least one positive weight required")
    draw = sample_float_weights(tuple(map(float, weights[support])), random_bits,
                                max_draws=max_draws)
    return replace(draw, token_index=int(support[draw.token_index]))
