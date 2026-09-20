"""Experimental batched binary64 tournament execution with small-support fallback.

Reference address/score laws and session lifecycle remain unchanged. Validate
bit tables once, preserve scalar math.fsum ordering and roundups each layer.
"""
import math
import numpy as np

from .._engine.legacy._impl.research.byte_trie_numeric import update, MIN_POSITIVE
from .._engine.legacy._impl.research.grouped_canonical_prototype import Prepared
from .._engine.legacy._impl.research.token_source_sparse_execution import (
    SparseTokenSourceSession, bit_table, nonzero_mass,
)

BATCH_MIN_CANDIDATES = 64


def update_layers(q, bits, diagnostics=None, *, observe=None):
    if (not isinstance(q, np.ndarray) or q.dtype != np.float64 or q.ndim != 1 or not q.size
            or not isinstance(bits, np.ndarray) or bits.dtype != np.int8 or bits.ndim != 2
            or bits.shape[1] != q.size or not bits.shape[0] or ((bits < 0) | (bits > 1)).any()):
        raise ValueError("invalid tournament inputs")
    if not np.isfinite(q).all() or (q < 0).any():
        raise ValueError("invalid tournament probability")
    r = q.copy()
    zero_masks = bits == 0
    floats = bits.astype(np.float64)
    with np.errstate(all="ignore"):
        for zero_mask, layer in zip(zero_masks, floats):
            total = math.fsum(r.tolist())
            if total <= 0 or abs(total - 1.) > 1e-12:
                raise ValueError("tournament probabilities must sum to one")
            zero_mass = math.fsum(r[zero_mask].tolist())
            weights = r * (zero_mass + layer * total)
            denominator = math.fsum(weights.tolist())
            if denominator <= 0 or not math.isfinite(denominator):
                raise ArithmeticError("invalid tournament normalization")
            output = weights / denominator
            floor = (r > 0) & (output == 0)
            floored = int(np.count_nonzero(floor))
            if floored:
                output[floor] = MIN_POSITIVE
                if diagnostics is not None:
                    diagnostics["branch_roundups"] = diagnostics.get("branch_roundups", 0) + floored
            r = output
            if observe is not None: observe(r, diagnostics)
    return r


def transform(q, profile, key, context, protected, counters, *, table=bit_table):
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
    labels = table(profile, key, context, distinct)
    if len(ids) >= BATCH_MIN_CANDIDATES:
        bits = np.array([labels[profile.classes[i]] for i in ids], dtype=np.int8).T
        r = update_layers(np.asarray(r, dtype=np.float64), bits, counters)
    else:
        for layer in range(profile.config.layers):
            r = update(r, [labels[profile.classes[i]][layer] for i in ids], counters)
    for i, p in zip(ids, r):
        scaled = mass * float(p)
        if scaled == 0.:
            scaled = MIN_POSITIVE
            counters["partition_roundups"] += 1
        out[i] = scaled
    return out


class BatchedTokenSourceSession(SparseTokenSourceSession):
    _bit_table = staticmethod(bit_table)

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
            out = transform(q, self.profile, self._key, self._context, protected, self._numeric_counters,
                            table=self._bit_table)
        else:
            # q already owns a snapshot of the caller's vector. The identity
            # branch does not write it, and Prepared still gets immutable bytes.
            out = q
        if not np.isfinite(out).all() or (out < 0).any():
            raise ArithmeticError("invalid transformed probability vector or support")
        support = out > 0
        # Reuse this exact mask for ordered mass, support validation and commit.
        # Keep every validation, including the full-vector finite/negative guard.
        if (abs(math.fsum(out[support]) - 1.) > 1e-12
                or not np.array_equal(support, q > 0)):
            raise ArithmeticError("invalid transformed probability vector or support")
        if protected and not np.array_equal(out[list(protected)], q[list(protected)]):
            raise ArithmeticError("protected source probability changed")
        self._last_decision = {"mode": mode, "protected_token_ids": protected,
                               "alignment_occurrences": occurrences}
        self._policy_counts[mode] += 1
        self._protected_candidates += len(protected)
        self._alignment_occurrences += occurrences
        self._pending = Prepared(self._steps, np.frombuffer(out.tobytes(), dtype=np.float64))
        self._support = support
        return self._pending
