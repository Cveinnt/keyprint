"""Explicit wider-head execution via an ordered sparse projection.

No frozen bound is edited. Select at most top_k finite logits in original-ID
order, apply the existing bounded filter, then restore full vocabulary indices.
Lower-ranked discarded logits cannot survive when a higher-ranked token fails
the monotone scaled-gap test. Ascending global IDs preserve all cutoff ties.
Only this experimental profile admits heads up to 262144; top_k remains bounded
by the reference's 151669-entry support. This is not detector qualification.
"""
from dataclasses import replace
import hashlib
from pathlib import Path

import numpy as np

from ..sampling import sparse_sample, sparse_softmax
from .._engine.research.keyprint_stable_support_filter_v3 import (
    FilterResult, MAX_MAPPED_VOCABULARY, stable_support_filter,
)

MAX_VOCABULARY = 262144


def identity():
    return {"implementation": "ordered-wide-head-projection-v1-experimental",
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "max_vocabulary": MAX_VOCABULARY, "max_support": MAX_MAPPED_VOCABULARY,
            "tie_order": "descending original float32 logit, ascending original token ID",
            "empirical_acceptance_transfers": False}


def support_filter(logits, *, temperature=.7, top_k=100, max_logit_gap=600.,
                   mapped_vocabulary_size=None):
    if not isinstance(logits, np.ndarray) or logits.dtype != np.float32:
        raise TypeError("model logits must be a NumPy float32 array; implicit casts are not allowed")
    if logits.ndim != 2 or logits.shape[0] != 1 or not 0 < logits.shape[1] <= MAX_VOCABULARY:
        raise ValueError("bounded single-row model logits required")
    width = logits.shape[1]
    mapped = width if mapped_vocabulary_size is None else mapped_vocabulary_size
    if type(mapped) is not int or not 1 <= mapped <= width:
        raise ValueError("mapped vocabulary must be an explicit prefix of the model head")
    if type(top_k) is not int or not 1 <= top_k <= MAX_MAPPED_VOCABULARY:
        raise ValueError("top_k must fit the reference support bound")
    if np.isnan(logits).any() or np.isposinf(logits).any():
        raise ValueError("NaN and positive infinity model logits are rejected")
    ids = np.flatnonzero(np.isfinite(logits[0, :mapped]))
    if not len(ids):
        raise ValueError("all mapped model logits are excluded")
    ranks = np.lexsort((ids, -logits[0, ids].astype(np.float64)))[:top_k]
    selected = np.sort(ids[ranks])
    packed = logits[:, selected]
    result = stable_support_filter(packed, temperature=temperature, top_k=top_k,
                                   max_logit_gap=max_logit_gap,
                                   mapped_vocabulary_size=len(selected))
    values = np.full((1, width), -np.inf, dtype=np.float64)
    values[:, selected] = result.filtered_logits
    values = np.frombuffer(values.tobytes(), dtype=np.float64).reshape(values.shape)
    return FilterResult(values,
        tuple(int(selected[i]) for i in result.admitted_token_ids),
        {**identity(), "vocabulary_size": width, "mapped_vocabulary_size": mapped,
         "reference_filter": result.identity},
        {"mapped_finite_count": len(ids), "projection_count": len(selected),
         "admitted_count": len(result.admitted_token_ids),
         "packed_reference": result.diagnostics, "input_was_modified": False})


def _vector(values):
    if (not isinstance(values, np.ndarray) or values.dtype != np.float64
            or values.ndim != 1 or not 0 < len(values) <= MAX_VOCABULARY):
        raise ValueError("expected a bounded float64 vocabulary vector")


def softmax(logits):
    _vector(logits)
    if np.isnan(logits).any() or np.isposinf(logits).any():
        raise ValueError("logits must be finite or negative infinity")
    ids = np.flatnonzero(np.isfinite(logits))
    if not 0 < len(ids) <= MAX_MAPPED_VOCABULARY:
        raise ValueError("finite support must fit the reference support bound")
    result = np.zeros(len(logits), dtype=np.float64)
    result[ids] = sparse_softmax(logits[ids])
    return result


def sample(weights, random_bits, *, max_draws=1024):
    _vector(weights)
    if not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("weights must be finite and nonnegative")
    ids = np.flatnonzero(weights > 0)
    if not 0 < len(ids) <= MAX_MAPPED_VOCABULARY:
        raise ValueError("positive support must fit the reference support bound")
    draw = sparse_sample(weights[ids], random_bits, max_draws=max_draws)
    return replace(draw, token_index=int(ids[draw.token_index]))
