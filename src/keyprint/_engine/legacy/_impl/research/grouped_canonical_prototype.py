"""Research-only single-response grouped sampler and raw event replay.

No model, tokenizer, sampling RNG, calibrated detector, channel router or HTTP API.
Caller must sample the prepared probabilities honestly, then commit exactly once.
No rollback/retry after cancellation. Never substitute for the frozen SDK.
"""
from dataclasses import dataclass, field
import hashlib
import hmac
import json
from threading import get_ident
from typing import Sequence
import numpy as np

VERSION = b"wm-probe/grouped-canonical/research-v1"
WHITESPACE = b" \t\r\n\f\v"


def pack(parts: Sequence[bytes]) -> bytes:
    """Injective length-prefixed encoding, including empty and nested contexts."""
    return len(parts).to_bytes(4, "big") + b"".join(
        len(part).to_bytes(8, "big") + part for part in parts)


@dataclass(frozen=True)
class Config:
    history: int = 4
    layers: int = 30
    max_steps: int = 4096

    def __post_init__(self):
        for name, low, high in (("history", 1, 64), ("layers", 1, 64), ("max_steps", 1, 65536)):
            value = getattr(self, name)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f"invalid {name}")


@dataclass(frozen=True, init=False)
class Profile:
    """Caller-declared token-byte mapping; None denotes excluded/formatting group.

    Non-ASCII bytes are unchanged. UTF-8 fragments stay bytes. Special token policy
    must be explicit in the supplied mapping; this module does not infer it.
    """
    config: Config
    classes: tuple[bytes | None, ...]
    tokenizer_identity: str
    digest: str
    _domain: bytes = field(repr=False)

    def __init__(self, token_bytes: Sequence[bytes | None], *, tokenizer_identity: str,
                 config: Config = Config()):
        if (not token_bytes or not isinstance(tokenizer_identity, str)
                or not tokenizer_identity or not isinstance(config, Config)):
            raise ValueError("mapping, tokenizer identity and config required")
        if any(piece is not None and type(piece) is not bytes for piece in token_bytes):
            raise TypeError("token pieces must be bytes or None")
        object.__setattr__(self, "config", config)
        object.__setattr__(self, "classes", tuple((piece.translate(None, WHITESPACE) or None)
                             if piece is not None else None for piece in token_bytes))
        object.__setattr__(self, "tokenizer_identity", tokenizer_identity)
        settings = json.dumps({"tokenizer": tokenizer_identity, **config.__dict__},
                              sort_keys=True, separators=(",", ":")).encode()
        body = pack([VERSION, settings, pack([piece or b"" for piece in self.classes])])
        object.__setattr__(self, "digest", hashlib.sha256(body).hexdigest())
        object.__setattr__(self, "_domain", pack([VERSION, bytes.fromhex(self.digest)]))

    def label(self, token_id: int):
        if type(token_id) is not int or not 0 <= token_id < len(self.classes):
            raise ValueError("token ID outside declared mapping")
        return self.classes[token_id]

    def bits(self, key: bytes, context: tuple[bytes, ...], label: bytes) -> tuple[int, ...]:
        if type(key) is not bytes or len(key) != 32:
            raise ValueError("key must contain exactly 32 bytes")
        if not label:
            raise ValueError("empty classes have no watermark bits")
        return tuple(hmac.digest(key, pack([self._domain, pack(context),
                                           layer.to_bytes(4, "big"), label]), "sha256")[0] & 1
                     for layer in range(self.config.layers))


@dataclass(frozen=True, eq=False)
class Prepared:
    index: int
    probabilities: np.ndarray = field(repr=False)


@dataclass(frozen=True)
class Event:
    context: tuple[bytes, ...]
    label: bytes
    eligible: bool
    bits: tuple[int, ...] | None


class Session:
    """Single-owner, one-response state; prepare -> sample externally -> commit."""
    def __init__(self, profile: Profile, key: bytes):
        if type(key) is not bytes or len(key) != 32:
            raise ValueError("key must contain exactly 32 bytes")
        if not isinstance(profile, Profile):
            raise TypeError("declared Profile required")
        self._profile, self._key = profile, key
        self._owner = get_ident()
        self._context: tuple[bytes, ...] = ()
        self._used: set[tuple[bytes, ...]] = set()
        self._pending: Prepared | None = None
        self._support = None
        self._steps = 0
        self._closed = False

    @property
    def profile(self):
        return self._profile

    def _ready(self):
        if get_ident() != self._owner:
            raise RuntimeError("session belongs to another thread")
        if self._closed:
            raise RuntimeError("session is closed")

    def prepare(self, probabilities: np.ndarray) -> Prepared:
        self._ready()
        if self._pending is not None:
            raise RuntimeError("commit or close the pending step first")
        if self._steps >= self.profile.config.max_steps:
            raise RuntimeError("step cap reached; context history is never evicted")
        if not isinstance(probabilities, np.ndarray) or probabilities.dtype != np.float64:
            raise TypeError("prototype requires NumPy float64 probabilities")
        q = probabilities.copy()
        if (q.shape != (len(self.profile.classes),) or not np.isfinite(q).all()
                or (q < 0).any() or abs(float(q.sum()) - 1) > 1e-12):
            raise ValueError("invalid base probability vector")
        if self._context not in self._used:
            indices = np.array([i for i, label in enumerate(self.profile.classes)
                                if label is not None and q[i] > 0], dtype=np.int64)
            if len(indices):
                mass = float(q[indices].sum())
                r = q[indices] / mass
                by_class = {self.profile.classes[i]: None for i in indices}
                for label in by_class:
                    by_class[label] = self.profile.bits(self._key, self._context, label)
                matrix = np.array([by_class[self.profile.classes[i]] for i in indices], dtype=np.float64)
                for layer in range(self.profile.config.layers):
                    g = matrix[:, layer]
                    g_mass = float(np.dot(r, g))
                    if not -1e-12 <= g_mass <= 1 + 1e-12:
                        raise ArithmeticError("tournament mass out of bounds")
                    r *= 1 + g - min(1.0, max(0.0, g_mass))
                q[indices] = mass * r
        if not np.isfinite(q).all() or (q < 0).any() or abs(float(q.sum()) - 1) > 1e-12:
            raise ArithmeticError("invalid transformed probability vector")
        # Immutable bytes-backed views prevent accidental mutation by the caller.
        exposed = np.frombuffer(q.tobytes(), dtype=np.float64)
        self._support = q > 0
        self._pending = Prepared(self._steps, exposed)
        return self._pending

    def commit(self, step: Prepared, token_id: int) -> Event | None:
        self._ready()
        if step is not self._pending or self._pending is None:
            raise ValueError("foreign, missing or stale prepared step")
        label = self.profile.label(token_id)
        if not self._support[token_id]:
            raise ValueError("committed token has zero prepared probability")
        event = None
        if label is not None:
            eligible = self._context not in self._used
            event = Event(self._context, label, eligible,
                          self.profile.bits(self._key, self._context, label) if eligible else None)
            self._used.add(self._context)
            self._context = (*self._context, label)[-self.profile.config.history:]
        self._steps += 1
        self._pending = self._support = None
        return event

    def close(self):
        self._ready()
        self._pending = self._support = None
        self._closed = True


def replay_events(profile: Profile, key: bytes, token_ids: Sequence[int]) -> tuple[Event, ...]:
    """Uncalibrated keyed events, not a detector decision or authenticity score.

    Input must be the declared tokenizer path. Equivalent rendered strings do not
    guarantee equivalent events. This intentionally exports no p-value/threshold.
    """
    if type(key) is not bytes or len(key) != 32:
        raise ValueError("key must contain exactly 32 bytes")
    if len(token_ids) > profile.config.max_steps:
        raise ValueError("replay exceeds the declared response step cap")
    context: tuple[bytes, ...] = ()
    used = set()
    events = []
    for token_id in token_ids:
        label = profile.label(token_id)
        if label is None:
            continue
        eligible = context not in used
        events.append(Event(context, label, eligible,
                            profile.bits(key, context, label) if eligible else None))
        used.add(context)
        context = (*context, label)[-profile.config.history:]
    return tuple(events)
