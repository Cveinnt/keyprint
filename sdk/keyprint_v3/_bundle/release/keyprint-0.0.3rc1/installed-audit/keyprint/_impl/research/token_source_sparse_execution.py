"""Exact source-session execution optimization; unchanged profile and scoring law.

Prepack repeated HMAC address prefixes and omit zero terms from validated
nonnegative mass sums. No model, key-file access or persistent bit cache.
The frozen reference remains untouched and is the parity oracle.
"""
import hmac
import math
import numpy as np

from .byte_trie_numeric import MIN_POSITIVE, update
from .grouped_canonical_prototype import Profile, Prepared, pack
from .token_source_policy import TokenSourceSession

REFERENCE_BITS = Profile.bits


def bit_table(profile, key, context, labels):
    # Preserve custom/fixture overrides rather than substituting their PRF.
    if getattr(profile.bits, "__func__", None) is not REFERENCE_BITS:
        return {label: profile.bits(key, context, label) for label in labels}
    context_bytes = pack(context)
    prefix = ((4).to_bytes(4, "big") + len(profile._domain).to_bytes(8, "big") + profile._domain
              + len(context_bytes).to_bytes(8, "big") + context_bytes + (4).to_bytes(8, "big"))
    prefixes = tuple(prefix + layer.to_bytes(4, "big") for layer in range(profile.config.layers))
    result = {}
    for label in labels:
        suffix = len(label).to_bytes(8, "big") + label
        result[label] = tuple(hmac.digest(key, head + suffix, "sha256")[0] & 1 for head in prefixes)
    return result


def transform(q, profile, key, context, protected, counters):
    protected = frozenset(protected)
    if any(type(i) is not int or not 0 <= i < len(q) for i in protected):
        raise ValueError("invalid protected token ID")
    ids = [int(i) for i in np.flatnonzero(q > 0)
           if int(i) not in protected and profile.classes[int(i)] is not None]
    distinct = dict.fromkeys(profile.classes[i] for i in ids)
    out = q.copy()
    if len(distinct) < 2:
        return out
    mass = math.fsum(float(q[i]) for i in ids)
    r = tuple(float(q[i]) / mass for i in ids)
    labels = bit_table(profile, key, context, distinct)
    for layer in range(profile.config.layers):
        r = update(r, [labels[profile.classes[i]][layer] for i in ids], counters)
    for i, p in zip(ids, r):
        scaled = mass * p
        if scaled == 0.:
            scaled = MIN_POSITIVE
            counters["partition_roundups"] += 1
        out[i] = scaled
    return out


def nonzero_mass(values):
    """Caller first verifies finite nonnegative entries; retain ascending order."""
    return math.fsum(values[values > 0])


class SparseTokenSourceSession(TokenSourceSession):
    def prepare(self, probabilities):
        self._ready()
        if self._pending is not None:
            raise RuntimeError("commit or close the pending step first")
        if self._steps >= self.profile.config.max_steps:
            raise RuntimeError("step cap reached; context history is never evicted")
        if not isinstance(probabilities, np.ndarray) or probabilities.dtype != np.float64:
            raise TypeError("prototype requires NumPy float64 probabilities")
        q = probabilities.copy()
        if (q.shape != (len(self.profile.classes),) or not np.isfinite(q).all()
                or (q < 0).any() or abs(nonzero_mass(q) - 1.) > 1e-12):
            raise ValueError("invalid base probability vector")
        mode, protected, occurrences = self._decision(q)
        if self._condition == "marked" and mode != "startup_ordinary" and self._context not in self._used:
            out = transform(q, self.profile, self._key, self._context, protected, self._numeric_counters)
        else:
            out = q.copy()
        if (not np.isfinite(out).all() or (out < 0).any()
                or abs(nonzero_mass(out) - 1.) > 1e-12 or not np.array_equal(out > 0, q > 0)):
            raise ArithmeticError("invalid transformed probability vector or support")
        if protected and not np.array_equal(out[list(protected)], q[list(protected)]):
            raise ArithmeticError("protected source probability changed")
        self._last_decision = {"mode": mode, "protected_token_ids": protected,
                               "alignment_occurrences": occurrences}
        self._policy_counts[mode] += 1
        self._protected_candidates += len(protected)
        self._alignment_occurrences += occurrences
        self._pending = Prepared(self._steps, np.frombuffer(out.tobytes(), dtype=np.float64))
        self._support = out > 0
        return self._pending
