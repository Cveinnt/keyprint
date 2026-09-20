"""Research-only model-centered bit bets, not a deployed detector.

At each fresh context/layer, residual = observed_bit - sum(p_label * bit_label).
For fixed key-independent text and predictions, an ideal independent fair PRF
gives each residual mean zero. Products of 1 + stake*residual then have mean
one across fresh addresses. This statement averages over ideal random keys;
it is not calibration for the fixed deployed HMAC key or key-dependent text.
The finite-precision implementation has no certified error-rate guarantee.

Uniformly average five whole-document betting processes. Do not select the
best stake, fit weights, change the cutoff, or omit unfavorable positions.
"""
import math

import numpy as np
from scipy.special import logsumexp

from surrogate_likelihood import dense_head
from keyprint._engine.legacy._impl.research.token_source_sparse_execution import bit_table

STAKES = (1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2)
LOG_CUTOFF = math.log(200.)


def layer_residuals(profile, key, context, token, probabilities):
    groups = {}
    for i in np.flatnonzero(probabilities):
        label = profile.classes[int(i)]
        if label is not None:
            groups.setdefault(label, []).append(float(probabilities[i]))
    masses = {label: math.fsum(parts) for label, parts in groups.items()}
    total = math.fsum(masses.values())
    label = profile.classes[token]
    if label not in masses or total <= 0:
        raise ValueError("Observed label must have positive markable support")
    bits = bit_table(profile, key, context, masses)
    weights = {label: mass / total for label, mass in masses.items()}
    expected = [math.fsum(weights[name] * bits[name][layer] for name in masses)
                for layer in range(profile.config.layers)]
    if any(not math.isfinite(v) or not -1e-12 <= v <= 1 + 1e-12 for v in expected):
        raise ArithmeticError("Predictive bit mass is not a probability")
    # Bound at most floating-point summation noise, not model probabilities.
    return [float(bit) - min(1., max(0., mean)) for bit, mean in zip(bits[label], expected, strict=True)]


def aggregate(terms):
    scores = [math.fsum(math.log1p(stake * value) for term in terms
                        for value in term["residuals"]) for stake in STAKES]
    return scores, float(logsumexp(scores) - math.log(len(STAKES)))


def score(profile, key, measurement):
    if type(key) is not bytes or len(key) != 32:
        raise ValueError("32-byte key required")
    ids, heads = measurement["token_ids"], measurement["heads"]
    if (not ids or len(ids) != len(heads) or len(ids) > profile.config.max_steps
            or any(type(i) is not int or not 0 <= i < len(profile.classes) for i in ids)):
        raise ValueError("Exact bounded token/head alignment required")
    context, used, terms = (), set(), []
    for index, (token, head) in enumerate(zip(ids, heads, strict=True)):
        probabilities = dense_head(head, len(profile.classes))
        label = profile.classes[token]
        reason, residuals = "scored", []
        if label is None:
            reason = "excluded_label"
        elif context in used:
            reason = "repeated_context"
        elif probabilities[token] == 0:
            reason = "outside_surrogate_support"
        else:
            residuals = layer_residuals(profile, key, context, token, probabilities)
        terms.append({"index": index, "token": token, "reason": reason, "residuals": residuals})
        if label is not None:
            used.add(context)
            context = (*context, label)[-profile.config.history:]
    components, value = aggregate(terms)
    return {"working_log_evidence": value, "component_log_evidence": components,
            "flagged": value >= LOG_CUTOFF, "terms": terms,
            "scored_events": sum(t["reason"] == "scored" for t in terms),
            "outside_support": sum(t["reason"] == "outside_surrogate_support" for t in terms),
            "calibrated": False}
