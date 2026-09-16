"""Bounded, single-owner byte-trie lifecycle and visible-text diagnostics.

Research hypothesis only. No model loader, population detector, or reset API.
Existing September 5 prototype stays unchanged. Python private attributes are an
API discipline, not a security boundary against a malicious in-process caller.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
from threading import get_ident

import numpy as np

from .byte_trie_tournament import Config, DOMAIN, KeyBits, State, advance, canonical, replay
from .byte_trie_numeric import KERNEL_VERSION, MIN_POSITIVE, transform
from .grouped_null_score import NullScore, tail


@dataclass(frozen=True)
class HostConfig:
    history: int = 16
    layers: int = 30
    max_steps: int = 512

    def __post_init__(self):
        Config(self.history, self.layers)
        if type(self.max_steps) is not int or not 1 <= self.max_steps <= 512:
            raise ValueError("byte-trie response cap must be in [1, 512]")


@dataclass(frozen=True)
class Profile:
    pieces: tuple
    config: HostConfig
    digest: str

    @classmethod
    def build(cls, pieces, config=HostConfig(), tokenizer_identity="supplied-fixture"):
        if type(config) is not HostConfig:
            raise TypeError("explicit HostConfig required")
        pieces = tuple(pieces)
        if not pieces:
            raise ValueError("empty token mapping")
        for piece in pieces:
            canonical(piece)  # Validate all candidates, including zero support.
            if piece is not None and len(piece) > 256:
                raise ValueError("candidate byte length exceeds bounded trie depth")
        mapping = hashlib.sha256()
        for piece in pieces:
            mapping.update(b"N" if piece is None else b"B" + len(piece).to_bytes(4, "big") + piece)
        spec = {"version": "byte-trie-visible-v3", "config": asdict(config),
                "numeric_kernel": KERNEL_VERSION,
                "positive_underflow_floor_hex": MIN_POSITIVE.hex(),
                "mapping_sha256": mapping.hexdigest(), "tokenizer_identity": tokenizer_identity,
                "prf_domain": DOMAIN.decode(), "normalization": "delete_ASCII_whitespace_9_13_32",
                "statistic": "all_unique_context_byte_layer_bits_binomial_diagnostic",
                "root_empty_mass": "copy_base_exactly_no_global_renormalization"}
        digest = hashlib.sha256(json.dumps(spec, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        return cls(pieces, config, digest)


@dataclass(frozen=True)
class Prepared:
    index: int
    probabilities: np.ndarray


def score_events(events, layers):
    """Ideal fresh random-function null only; HMAC assumption is not a proof."""
    seen, eligible, ones = set(), 0, 0
    for event in events:
        if not event.eligible:
            if event.bits is not None:
                raise ValueError("ineligible byte event has bits")
            continue
        if event.context in seen or event.bits is None or len(event.bits) != layers:
            raise ValueError("invalid/repeated eligible byte context")
        if any(type(b) is not int or b not in (0, 1) for b in event.bits):
            raise ValueError("nonbinary byte event")
        seen.add(event.context)
        eligible += 1
        ones += sum(event.bits)
    if not eligible:
        return NullScore(0, 0, 0, None, False)
    p, floor = tail(ones, eligible * layers)
    return NullScore(eligible, eligible * layers, ones, p, floor)


def inspect_text(text, key, config=HostConfig()):
    if type(text) is not str:
        raise TypeError("visible text must be str")
    raw = text.encode("utf-8", "strict")
    if len(raw) > 2 * 1024 * 1024:
        raise ValueError("visible text exceeds byte cap")
    state, events = replay((raw,), config, KeyBits(key, config))
    return state, events, score_events(events, config.layers)


class Session:
    def __init__(self, profile, key, *, condition):
        if type(profile) is not Profile or condition not in ("ordinary", "marked"):
            raise ValueError("explicit byte profile and ordinary/marked condition required")
        self._profile, self._condition = profile, condition
        self._bits = KeyBits(key, profile.config)
        self._owner, self._terminal = get_ident(), False
        self._state, self._events, self._ids = State(), [], []
        self._pending = None
        self._numeric_counters = {"branch_roundups": 0, "token_roundups": 0}

    @property
    def profile(self):
        return self._profile

    @property
    def state(self):
        self._owner_check()
        return self._state

    @property
    def events(self):
        self._owner_check()
        return tuple(self._events)

    @property
    def token_ids(self):
        self._owner_check()
        return tuple(self._ids)

    @property
    def numeric_counters(self):
        self._owner_check()
        return dict(self._numeric_counters)

    def _owner_check(self):
        if get_ident() != self._owner:
            raise RuntimeError("byte-trie session belongs to another thread")

    def _ready(self):
        self._owner_check()
        if self._terminal:
            raise RuntimeError("response is terminal; no retry or reset")

    def close(self):
        self._owner_check()
        self._terminal, self._pending = True, None
        self._bits = None

    def _prepare_probabilities(self, q):
        """Numeric policy hook; lifecycle and immutable preparation stay shared."""
        if self._condition == "ordinary":
            return q
        support = np.flatnonzero(q > 0)
        base = {int(i): float(q[i]) for i in support}
        pieces = {int(i): self.profile.pieces[i] for i in support}
        diagnostics = {"branch_roundups": 0, "token_roundups": 0}
        out = transform(base, pieces, self._state, self.profile.config, self._bits, diagnostics)
        for i, probability in out.items():
            # Algebraically identical; avoid a root p/total*total round trip.
            q[i] = base[i] if not canonical(pieces[i]) else probability
        for name, value in diagnostics.items():
            self._numeric_counters[name] += value
        return q

    def prepare(self, probabilities):
        self._ready()
        try:
            if self._pending is not None:
                raise RuntimeError("commit or close the pending step first")
            if len(self._ids) >= self.profile.config.max_steps:
                raise RuntimeError("step cap reached; context history is never evicted")
            if not isinstance(probabilities, np.ndarray) or probabilities.dtype != np.float64:
                raise TypeError("byte-trie host requires NumPy float64 probabilities")
            q = probabilities.copy()
            if (q.shape != (len(self.profile.pieces),) or not np.isfinite(q).all()
                    or (q < 0).any() or abs(float(q.sum()) - 1.) > 1e-12):
                raise ValueError("invalid base probability vector")
            q = self._prepare_probabilities(q)
            if not np.isfinite(q).all() or (q < 0).any() or abs(float(q.sum()) - 1.) > 1e-10:
                raise ArithmeticError("invalid byte-trie output distribution")
            self._pending = Prepared(len(self._ids), np.frombuffer(q.tobytes(), dtype=np.float64))
            return self._pending
        except Exception:
            self.close()
            raise

    def commit(self, prepared, token_id):
        self._ready()
        try:
            if prepared is not self._pending or prepared is None:
                raise RuntimeError("foreign, stale or absent preparation")
            if (type(token_id) is not int or not 0 <= token_id < len(self.profile.pieces)
                    or prepared.probabilities[token_id] <= 0):
                raise ValueError("selected token is outside prepared positive support")
            self._ids.append(token_id)  # Retain consumed sample even if advancing fails.
            self._pending = None
            self._state, added = advance(self._state, self.profile.pieces[token_id],
                                         self.profile.config, self._bits)
            self._events.extend(added)
            return added
        except Exception:
            self.close()
            raise
