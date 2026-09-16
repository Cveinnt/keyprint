"""Explicit source-copy hypothesis over the shared byte-trie lifecycle."""
from dataclasses import dataclass, field, replace
import hashlib
import json
import math

import numpy as np

from .byte_trie_numeric import MIN_POSITIVE, transform
from .byte_trie_session import Session
from .byte_trie_tournament import canonical

MATCH_BYTES = 24
STARTUP_BYTES = 96
MAX_SOURCE_BYTES = 65536


def digest_spec(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def policy_spec():
    return {"version": "byte-trie-source-continuations-v1", "match_bytes": MATCH_BYTES,
            "startup_bytes": STARTUP_BYTES, "max_source_bytes": MAX_SOURCE_BYTES,
            "alignment": "all_exact_overlapping_canonical_byte_suffix_occurrences",
            "protection": "every_supported_whole_piece_source_continuation",
            "partition": "direct_fsum_mass_conditional_existing_kernel_scale_back",
            "scale_underflow": MIN_POSITIVE.hex(), "empty_control_mass": "copy_base_exactly",
            "source_canonicalization": "delete_ASCII_whitespace_9_13_32_no_Unicode_normalization",
            "purposes": ["general", "proofread", "transform"], "automatic_routing": False,
            "detector": "unchanged_all_eligible_byte_diagnostic_no_source_filter"}


@dataclass(frozen=True)
class SourceRequest:
    purpose: str = "general"
    source_text: str | None = field(default=None, repr=False)
    canonical_source: bytes = field(init=False, repr=False)
    source_sha256: str | None = field(init=False)

    def __post_init__(self):
        if self.purpose not in ("general", "proofread", "transform"):
            raise ValueError("explicit general, proofread or transform purpose required")
        if self.purpose == "general" and self.source_text is not None:
            raise ValueError("general purpose does not accept source text")
        if self.source_text is not None and type(self.source_text) is not str:
            raise TypeError("source text must be str or None")
        raw = self.source_text.encode("utf-8", "strict") if self.source_text is not None else b""
        if len(raw) > MAX_SOURCE_BYTES:
            raise ValueError("source exceeds 65536-byte cap")
        clean = canonical(raw)
        if self.purpose == "proofread" and not clean:
            raise ValueError("proofread requires nonempty canonical source evidence")
        object.__setattr__(self, "canonical_source", clean)
        object.__setattr__(self, "source_sha256",
                           hashlib.sha256(raw).hexdigest() if self.source_text is not None else None)

    def receipt(self):
        return {"purpose": self.purpose, "source_sha256": self.source_sha256,
                "source_canonical_bytes": len(self.canonical_source)}


def protect_partition(q, pieces, state, config, bits, protected, counters):
    """Preserve protected and empty/control probabilities; transform alternatives.

Caller validates the base distribution through Session.prepare. This helper
keeps every input-zero coordinate zero and never globally renormalizes output.
"""
    protected = set(protected)
    if any(type(i) is not int or not 0 <= i < len(q) for i in protected):
        raise ValueError("invalid protected candidate ID")
    active = [int(i) for i in np.flatnonzero(q > 0)
              if int(i) not in protected and canonical(pieces[int(i)])]
    out = q.copy()
    if len(active) < 2:
        return out
    mass = math.fsum(float(q[i]) for i in active)
    probabilities = {i: float(q[i]) / mass for i in active}
    mapped = {i: pieces[i] for i in active}
    changed = transform(probabilities, mapped, state, config, bits, counters)
    for i in active:
        scaled = changed[i] * mass
        if scaled == 0.:
            scaled = MIN_POSITIVE
            counters["partition_roundups"] = counters.get("partition_roundups", 0) + 1
        out[i] = scaled
    return out


class SourceSession(Session):
    def __init__(self, profile, key, *, condition, request=SourceRequest()):
        if type(request) is not SourceRequest:
            raise TypeError("explicit SourceRequest required")
        super().__init__(profile, key, condition=condition)
        self._base_profile_sha256 = profile.digest
        self._profile = replace(profile, digest=digest_spec({"base_profile": profile.digest,
                                                           "source_policy": policy_spec()}))
        self._request = request
        self._source_output = b""
        self._alignment_seen = self._startup_exhausted = False
        self._policy_counts = {"startup_ordinary": 0, "source_protected": 0, "full_mark": 0}
        self._protected_candidates = self._alignment_occurrences = 0
        self._max_protected_error = 0.
        self._last_decision = None
        self._numeric_counters["partition_roundups"] = 0

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
        protected = []
        for i in np.flatnonzero(q > 0):
            piece = canonical(self.profile.pieces[int(i)])
            if piece and any(source.startswith(piece, end) for end in ends):
                protected.append(int(i))
        return ("source_protected" if protected else "full_mark"), tuple(protected), len(ends)

    def _prepare_probabilities(self, q):
        mode, protected, occurrences = self._decision(q)
        self._last_decision = {"mode": mode, "protected_token_ids": protected,
                               "alignment_occurrences": occurrences}
        self._policy_counts[mode] += 1
        self._protected_candidates += len(protected)
        self._alignment_occurrences += occurrences
        if self._condition == "ordinary" or mode == "startup_ordinary":
            return q
        if mode == "full_mark":
            return super()._prepare_probabilities(q)
        out = protect_partition(q, self.profile.pieces, self._state, self.profile.config,
                                self._bits, protected, self._numeric_counters)
        if protected:
            error = max(abs(float(out[i]) - float(q[i])) for i in protected)
            self._max_protected_error = max(self._max_protected_error, error)
            if error != 0.:
                raise ArithmeticError("protected source probability changed")
        return out

    def commit(self, prepared, token_id):
        events = super().commit(prepared, token_id)
        self._source_output += canonical(self.profile.pieces[token_id])
        return events

    @property
    def last_decision(self):
        self._owner_check()
        return dict(self._last_decision) if self._last_decision is not None else None

    def source_receipt(self):
        self._owner_check()
        return {"base_profile_sha256": self._base_profile_sha256,
                "source_policy_sha256": digest_spec(policy_spec()), **self._request.receipt(),
                "decision_counts": dict(self._policy_counts),
                "protected_candidate_count": self._protected_candidates,
                "alignment_occurrence_count": self._alignment_occurrences,
                "max_protected_probability_error": self._max_protected_error,
                "alignment_seen": self._alignment_seen, "startup_exhausted": self._startup_exhausted,
                "automatic_routing": False, "source_filtered_detection": False}
