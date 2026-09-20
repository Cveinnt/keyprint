"""Exact partition execution of the frozen shared support-filter law.

The returned policy identity names the unchanged reference law. This module's
implementation hash is separately bound into the native execution identity.
Validation, arithmetic, diagnostics and immutable output match the reference;
only the selection algorithm differs. The frozen engine is never patched.
"""
import math
import numpy as np

from .._engine.research.keyprint_stable_support_filter_v3 import (
    FilterResult, filter_identity, DEFAULT_MAX_LOGIT_GAP,
    MAX_MODEL_VOCABULARY, MAX_MAPPED_VOCABULARY,
)


def select_top_k(candidate_ids, scores, top_k, native=None):
    """Same descending-score/ascending-ID order on validated filter candidates.

    The filter supplies strictly increasing candidate IDs and finite float64
    scores converted exactly from float32. This helper does not change filtering,
    temperature arithmetic, gap boundaries or the final immutable probability
    support. It only avoids sorting discarded candidates.
    """
    if native is not None:
        from keyprint_native import NativePRF
        if type(native) is not NativePRF:
            raise TypeError('exact native backend required')
    if len(scores) <= top_k:
        return np.lexsort((candidate_ids, -scores))
    if np.all(scores == scores[0]):
        return np.arange(top_k)
    if native is not None and top_k <= 512 and len(scores) >= 1024:
        # These binary64 values came exactly from binary32 model scores.
        # Candidate IDs are ascending, so original-index ties preserve ID ties.
        return np.asarray(native.select_indices(scores.astype('<f4').tobytes(), top_k), dtype=np.int64)
    threshold = np.partition(scores, len(scores) - top_k)[len(scores) - top_k]
    above = np.flatnonzero(scores > threshold)
    tied = np.flatnonzero(scores == threshold)[:top_k - len(above)]
    selected = np.concatenate((above, tied))
    return selected[np.lexsort((candidate_ids[selected], -scores[selected]))]


def partition_support_filter(logits, *, temperature=.7, top_k=100,
                          max_logit_gap=DEFAULT_MAX_LOGIT_GAP,
                          mapped_vocabulary_size=None, native=None):
    """Define explicit post-filter support identically for both generation arms.

    A mapping argument admits a prefix of the model head. By default, all columns
    up to the explicit maximum mapped size are admitted; any later model padding
    is excluded independently of its score. Callers with a different mapping must
    not use this prefix-policy prototype without a separately specified adapter.

    Temperature accepts every finite positive Python float/int representable as
    binary64. The maximum scaled gap must lie in (0,600]. This conservative range
    keeps all retained base softmax probabilities normal even at the full mapped
    vocabulary bound. Branch rounding/floors in a later watermark transform are
    a separate contract; this filter makes no exact-real probability claim.
    """
    if not isinstance(logits, np.ndarray) or logits.dtype != np.float32:
        raise TypeError("model logits must be a NumPy float32 array; implicit casts are not allowed")
    if logits.ndim != 2 or logits.shape[0] != 1 or not 0 < logits.shape[1] <= MAX_MODEL_VOCABULARY:
        raise ValueError("bounded single-row model logits required")
    if type(temperature) not in (float, int):
        raise ValueError("temperature must be finite and positive")
    try:
        temperature = float(temperature)
    except OverflowError:
        raise ValueError("temperature must be finite and positive") from None
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    if type(max_logit_gap) not in (float, int):
        raise ValueError("maximum scaled logit gap must be in (0,600]")
    try:
        max_logit_gap = float(max_logit_gap)
    except OverflowError:
        raise ValueError("maximum scaled logit gap must be in (0,600]") from None
    if not math.isfinite(max_logit_gap) or not 0 < max_logit_gap <= 600:
        raise ValueError("maximum scaled logit gap must be in (0,600]")
    if type(top_k) is not int or not 1 <= top_k <= MAX_MAPPED_VOCABULARY:
        raise ValueError("top_k must be a positive integer within the mapped vocabulary bound")
    width = logits.shape[1]
    mapped = min(width, MAX_MAPPED_VOCABULARY) if mapped_vocabulary_size is None else mapped_vocabulary_size
    if type(mapped) is not int or not 1 <= mapped <= min(width, MAX_MAPPED_VOCABULARY):
        raise ValueError("mapped vocabulary must be a valid bounded prefix of the model head")
    # Invalid model output is rejected even outside the mapped prefix.
    if np.isnan(logits).any() or np.isposinf(logits).any():
        raise ValueError("NaN and positive infinity model logits are rejected")
    row = logits[0].astype(np.float64)
    finite_ids = np.flatnonzero(np.isfinite(row[:mapped]))
    if not len(finite_ids):
        raise ValueError("all mapped model logits are excluded")
    maximum = float(np.max(row[finite_ids]))
    losses = maximum - row[finite_ids]
    assert np.isfinite(losses).all() and (losses >= 0).all()
    # float32 loss is at most ~6.8e38. Multiplication may intentionally become
    # +inf for enormous temperature; every finite loss then passes this *coarse*
    # gate. For tiny temperature, rejecting before division prevents overflow.
    # A final rounded quotient can equal G even when the exact quotient is
    # slightly larger. Expand G upward, then round the product bound upward too;
    # a rounded-down G*T must not remove a token whose final quotient equals G.
    raw_gap = math.nextafter(math.nextafter(max_logit_gap, math.inf) * temperature, math.inf)
    possible = losses <= raw_gap
    candidate_ids, candidate_losses = finite_ids[possible], losses[possible]
    scaled_losses = candidate_losses / temperature
    # The final inclusive comparison defines the exact implemented gap boundary;
    # multiplying by temperature above can round at subnormal boundaries.
    inside = np.isfinite(scaled_losses) & (scaled_losses <= max_logit_gap)
    candidate_ids, scaled_losses = candidate_ids[inside], scaled_losses[inside]
    if not len(candidate_ids):
        raise ArithmeticError("the maximum token must survive stable support filtering")
    order = select_top_k(candidate_ids, row[candidate_ids], top_k, native=native)
    selected, scaled = candidate_ids[order], scaled_losses[order]
    out = np.full((1, width), -np.inf, dtype=np.float64)
    out[0, selected] = -scaled
    assert np.isfinite(out[0, selected]).all()
    assert np.max(out[0, selected]) == 0 and np.min(out[0, selected]) >= -max_logit_gap
    # Immutable backing prevents mutation without also explicitly making a copy.
    out = np.frombuffer(out.tobytes(), dtype=np.float64).reshape(out.shape)
    admitted = tuple(sorted(map(int, selected)))
    capacity = min(top_k, mapped)
    return FilterResult(
        filtered_logits=out,
        admitted_token_ids=admitted,
        identity=filter_identity(temperature=temperature, top_k=top_k, max_logit_gap=max_logit_gap,
                                 vocabulary_size=width, mapped_vocabulary_size=mapped),
        diagnostics={
            "input_finite_count": int(np.isfinite(row).sum()),
            "mapped_finite_count": len(finite_ids),
            "excluded_unmapped_finite_count": int(np.isfinite(row[mapped:]).sum()),
            "excluded_by_scaled_gap_count": len(finite_ids) - len(candidate_ids),
            "excluded_by_top_k_count": max(0, len(candidate_ids) - top_k),
            "admitted_count": len(admitted),
            # exp(-600) > 2**-866; K <= next power of two. This conservative
            # real-arithmetic bound is exactly representable, unlike an exp()
            # approximation which might round above its mathematical value.
            "retained_real_softmax_probability_lower_bound": math.ldexp(1., -866 - (capacity - 1).bit_length()),
            "retained_real_softmax_lower_bound_formula": "2**(-866-ceil(log2(K))), K=min(top_k,mapped_vocabulary_size)",
            "tighter_real_probability_lower_bound_approx": math.exp(-max_logit_gap) / capacity,
            "bound_is_exact_real_formula_not_float_error_certificate": True,
            "input_was_modified": False,
            "no_empirical_acceptance": True,
        },
    )
