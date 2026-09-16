"""Grouped token tournament plus explicit source-copy policy; new PRF profile."""
import hashlib
import math

import numpy as np

from .byte_trie_numeric import KERNEL_VERSION, MIN_POSITIVE, update
from .byte_trie_source_policy import SourceRequest, MATCH_BYTES, STARTUP_BYTES, digest_spec
from .byte_trie_tournament import canonical
from .grouped_canonical_prototype import Config, Prepared, Profile, Session
from .grouped_tokenizer_binding import Binding


def policy_spec():
    return {"version": "token-source-tournament-v1", "numeric_kernel": KERNEL_VERSION,
        "grouping": "canonical_nonempty_token_bytes_delete_ASCII_whitespace",
        "repeated_context": "identity", "single_distinct_label": "identity",
        "source_match_bytes": MATCH_BYTES, "startup_bytes": STARTUP_BYTES,
        "alignment": "all_exact_overlapping_canonical_byte_suffix_occurrences",
        "protection": "all_supported_whole_piece_continuations",
        "partition": "fsum_mass_conditional_tournament_scale_back",
        "partition_roundup": MIN_POSITIVE.hex(), "excluded_mass": "copy_base_exactly",
        "purpose_routing": "explicit_general_proofread_transform",
        "text_score": "all_eligible_grouped_events_even_when_generation_path_differs"}


class TokenSourceBinding(Binding):
    def __init__(self, config=Config(max_steps=512)):
        super().__init__(config)
        self.base_profile_sha256 = self.profile.digest
        self._profile = Profile(self.token_bytes,
            tokenizer_identity=self.profile.tokenizer_identity + ":" + digest_spec(policy_spec()),
            config=config)


def transform(q, profile, key, context, protected, counters):
    """Only conditional unprotected, nonempty-label mass enters the tournament."""
    protected = frozenset(protected)
    if any(type(i) is not int or not 0 <= i < len(q) for i in protected):
        raise ValueError("invalid protected token ID")
    ids = [int(i) for i in np.flatnonzero(q > 0)
           if int(i) not in protected and profile.classes[int(i)] is not None]
    labels = dict.fromkeys(profile.classes[i] for i in ids)
    out = q.copy()
    if len(labels) < 2:
        return out
    mass = math.fsum(float(q[i]) for i in ids)
    r = tuple(float(q[i]) / mass for i in ids)
    for label in labels:
        labels[label] = profile.bits(key, context, label)
    for layer in range(profile.config.layers):
        r = update(r, [labels[profile.classes[i]][layer] for i in ids], counters)
    for i, p in zip(ids, r):
        scaled = mass * p
        if scaled == 0.:
            scaled = MIN_POSITIVE
            counters["partition_roundups"] += 1
        out[i] = scaled
    return out


class TokenSourceSession(Session):
    def __init__(self, profile, key, *, condition, request=SourceRequest()):
        if condition not in ("ordinary", "marked") or type(condition) is not str:
            raise ValueError("condition must be ordinary or marked")
        if type(request) is not SourceRequest:
            raise TypeError("explicit SourceRequest required")
        super().__init__(profile, key)
        self._condition, self._request = condition, request
        self._source_output = b""
        self._alignment_seen = self._startup_exhausted = False
        self._policy_counts = dict.fromkeys(("startup_ordinary", "source_protected", "full_mark"), 0)
        self._numeric_counters = {"branch_roundups": 0, "partition_roundups": 0}
        self._last_decision = None
        self._protected_candidates = self._alignment_occurrences = 0

    def _owner_check(self):
        # Receipts remain readable after close, but only by their owner.
        from threading import get_ident
        if get_ident() != self._owner:
            raise RuntimeError("session belongs to another thread")

    def _decision(self, q):
        if self._request.purpose != "proofread":
            return "full_mark", (), 0
        source, generated = self._request.canonical_source, self._source_output
        ends = []
        if len(generated) >= MATCH_BYTES:
            suffix, start = generated[-MATCH_BYTES:], 0
            while (found := source.find(suffix, start)) >= 0:
                ends.append(found + MATCH_BYTES)
                start = found + 1
        if len(generated) >= STARTUP_BYTES:
            self._startup_exhausted = True
        if not ends:
            if not self._alignment_seen and not self._startup_exhausted:
                return "startup_ordinary", (), 0
            return "full_mark", (), 0
        self._alignment_seen = True
        protected = tuple(int(i) for i in np.flatnonzero(q > 0)
            if self.profile.classes[int(i)] and
            any(source.startswith(self.profile.classes[int(i)], end) for end in ends))
        return ("source_protected" if protected else "full_mark"), protected, len(ends)

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
                or (q < 0).any() or abs(math.fsum(q) - 1.) > 1e-12):
            raise ValueError("invalid base probability vector")
        mode, protected, occurrences = self._decision(q)
        if self._condition == "marked" and mode != "startup_ordinary" and self._context not in self._used:
            out = transform(q, self.profile, self._key, self._context, protected, self._numeric_counters)
        else:
            out = q.copy()
        if (not np.isfinite(out).all() or (out < 0).any()
                or abs(math.fsum(out) - 1.) > 1e-12 or not np.array_equal(out > 0, q > 0)):
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

    def commit(self, prepared, token_id):
        event = super().commit(prepared, token_id)
        self._source_output += self.profile.classes[token_id] or b""
        return event

    @property
    def last_decision(self):
        self._owner_check()
        return dict(self._last_decision) if self._last_decision is not None else None

    def source_receipt(self):
        self._owner_check()
        return {"source_policy_sha256": digest_spec(policy_spec()), **self._request.receipt(),
            "decision_counts": dict(self._policy_counts),
            "protected_candidate_count": self._protected_candidates,
            "alignment_occurrence_count": self._alignment_occurrences,
            "alignment_seen": self._alignment_seen, "startup_exhausted": self._startup_exhausted,
            "numeric_counters": dict(self._numeric_counters),
            "automatic_routing": False, "source_filtered_detection": False}
