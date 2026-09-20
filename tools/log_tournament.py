"""Log-space ideal grouped tournament for a research detector.

No generation mutation, probability flooring or calibrated-error guarantee.
Tiny probabilities remain represented as finite logs through all layers.
"""
import math
import numpy as np
from scipy.special import logsumexp

from keyprint._engine.legacy._impl.research.token_source_sparse_execution import bit_table


def token_log_ratio(p, profile, key, context, token):
    if (not isinstance(p, np.ndarray) or p.dtype != np.float64
            or p.shape != (len(profile.classes),) or not np.isfinite(p).all()
            or (p < 0).any() or abs(math.fsum(p[p > 0]) - 1.) > 1e-12
            or type(token) is not int or not 0 <= token < len(p) or p[token] <= 0):
        raise ValueError("Normalized base probabilities and a supported observation required")
    if profile.classes[token] is None:
        return 0.
    ids = [int(i) for i in np.flatnonzero(p > 0) if profile.classes[int(i)] is not None]
    labels = dict.fromkeys(profile.classes[i] for i in ids)
    if len(labels) < 2:
        return 0.
    bits = bit_table(profile, key, context, labels)
    raw = np.log(p[ids])
    log_weights = raw - logsumexp(raw)
    position = ids.index(token)
    initial = float(log_weights[position])
    for layer in range(profile.config.layers):
        g = np.asarray([bits[profile.classes[i]][layer] for i in ids], dtype=bool)
        if g.all() or not g.any():
            continue
        log_total = float(logsumexp(log_weights))
        log_zero = float(logsumexp(log_weights[~g]))
        factors = np.where(g, np.logaddexp(log_zero, log_total), log_zero)
        updated = log_weights + factors
        log_weights = updated - logsumexp(updated)
        if not np.isfinite(log_weights).all():
            raise ArithmeticError("Nonfinite log-space tournament weights")
    return float(log_weights[position]) - initial


def mixture_log_factor(log_ratio, mixture=.5):
    if not math.isfinite(log_ratio) or mixture != .5:
        raise ValueError("Finite log ratio and the frozen half-mixture required")
    return float(np.logaddexp(math.log(.5), math.log(.5) + log_ratio))
