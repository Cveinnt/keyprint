"""Unintegrated shared ordinary/marked numerical support filter proposal.

Input model logits must already be NumPy float32, shape (1, vocabulary_size).
No implicit narrowing cast, condition flag, key, model or sampling RNG is used.
Float64 centering safely subtracts any two finite float32 inputs. A documented
scaled-gap filter removes extreme tails before division by temperature; no finite
admitted value can silently overflow to an exclusion marker. All returned finite
logits lie in [-max_logit_gap, 0], with at least one exact zero.

This creates a NEW ordinary filtering policy and changes supported probabilities.
It is not integrated into v2, not a frozen SDK change and not an A10 acceptance.
"""
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

import numpy as np


VERSION = "keyprint-shared-stable-support-filter-v3-prototype-2026-09-09"
MAX_MODEL_VOCABULARY = 151936
MAX_MAPPED_VOCABULARY = 151669
DEFAULT_MAX_LOGIT_GAP = 600.0


@dataclass(frozen=True)
class FilterResult:
    filtered_logits: np.ndarray
    admitted_token_ids: tuple[int, ...]
    identity: dict
    diagnostics: dict


def filter_identity(*, temperature, top_k, max_logit_gap, vocabulary_size, mapped_vocabulary_size):
    specification = {
        "version": VERSION,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "input_dtype": "float32_only_no_implicit_cast",
        "output_dtype": "float64",
        "centering": "float64_max_minus_float64_input_before_temperature_division",
        "support_order": "mapped_prefix_then_finite_then_scaled_gap_then_top_k",
        "max_scaled_gap_comparison": "inclusive_after_float64_division",
        "coarse_gap_gate": "nextafter(nextafter(gap,+inf)*temperature,+inf); final division is authoritative",
        "top_k_tie_break": "descending_original_float32_logit_then_ascending_token_id",
        "temperature": float(temperature),
        "top_k": top_k,
        "max_logit_gap": float(max_logit_gap),
        "vocabulary_size": vocabulary_size,
        "mapped_vocabulary_size": mapped_vocabulary_size,
        "ordinary_marked_same_function": True,
        "watermark_dependent_filtering": False,
        "gap_tail_floor": False,
        "integration_status": "standalone_unintegrated_prototype",
    }
    encoded = json.dumps(specification, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return {"filter_profile_sha256": hashlib.sha256(encoded).hexdigest(), "specification": specification}


def stable_support_filter(logits, *, temperature=.7, top_k=100,
                          max_logit_gap=DEFAULT_MAX_LOGIT_GAP,
                          mapped_vocabulary_size=None):
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
    order = np.lexsort((candidate_ids, -row[candidate_ids]))[:top_k]
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
