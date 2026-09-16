"""Research-only causal canonical-byte trie tournament, not a release profile.

Preserve token-ending/empty/control mass; transform only competing next bytes.
Output is still a probability vector over existing tokens. No model, decoder,
token sampler, population detector, or Anthropic attribution. Fraction inputs
support exact finite checks; floating-point execution has ordinary roundoff.
"""
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import hmac
import math

SPACE = b" \t\r\n\f\v"
DOMAIN = b"wm-probe/byte-trie/research-v1"


def canonical(piece):
    if piece is None:
        return b""
    if type(piece) is not bytes:
        raise TypeError("token piece must be bytes or None")
    return piece.translate(None, SPACE)


@dataclass(frozen=True)
class Config:
    history: int = 16
    layers: int = 30

    def __post_init__(self):
        if type(self.history) is not int or not 1 <= self.history <= 64:
            raise ValueError("invalid byte history")
        if type(self.layers) is not int or not 1 <= self.layers <= 64:
            raise ValueError("invalid layer count")


class KeyBits:
    def __init__(self, key, config):
        if type(key) is not bytes or len(key) != 32:
            raise ValueError("32-byte research key required")
        self.key, self.config = key, config

    def __call__(self, context, byte, layer):
        payload = (DOMAIN + bytes([self.config.history, self.config.layers,
                                   len(context), layer, byte]) + context)
        return hmac.digest(self.key, payload, "sha256")[0] & 1


@dataclass(frozen=True)
class State:
    context: bytes = b""
    used: frozenset = frozenset()


@dataclass(frozen=True)
class Event:
    context: bytes
    byte: int
    eligible: bool
    bits: tuple | None


def advance(state, piece, config, bits):
    context, used, events = state.context, set(state.used), []
    for byte in canonical(piece):
        eligible = context not in used
        values = tuple(bits(context, byte, d) for d in range(config.layers)) if eligible else None
        if values is not None and any(type(x) is not int or x not in (0, 1) for x in values):
            raise ValueError("nonbinary oracle result")
        events.append(Event(context, byte, eligible, values))
        used.add(context)
        context = (context + bytes([byte]))[-config.history:]
    return State(context, frozenset(used)), tuple(events)


def replay(pieces, config, bits):
    state, events = State(), []
    for piece in pieces:
        state, added = advance(state, piece, config, bits)
        events.extend(added)
    return state, tuple(events)


def update(q, bits):
    """Single public SynthID tournament equation; checked against pinned upstream."""
    if len(q) != len(bits) or any(type(g) is not int or g not in (0, 1) for g in bits):
        raise ValueError("invalid tournament inputs")
    mean = sum(p * g for p, g in zip(q, bits))
    return tuple(p * (1 + g - mean) for p, g in zip(q, bits))


def transform(probabilities, pieces, state, config, bits):
    """Pure preparation. Caller may commit only a genuinely selected token.

At each trie node, mass ending at that node remains unchanged conditional on
arrival. Remaining mass is distributed among byte children by the tournament.
Descendant transforms do not change aggregate child mass, preventing candidate
lookahead leakage into ancestor selection. No end-of-token mark is scored.
Repeated contexts bypass within each candidate path as well as prior output.
"""
    if not isinstance(state, State) or len(state.context) > config.history:
        raise ValueError("invalid byte-trie state")
    if not probabilities or set(probabilities) - set(pieces):
        raise ValueError("every candidate needs an explicit byte mapping")
    if any(type(k) is not int or k < 0 for k in probabilities):
        raise ValueError("candidate IDs must be nonnegative integers")
    values = tuple(probabilities.values())
    exact = all(isinstance(p, Fraction) for p in values)
    if any(not math.isfinite(p) or p < 0 for p in values):
        raise ValueError("invalid base weights")
    total = sum(values)
    if (total != 1 if exact else abs(total - 1) > 1e-12):
        raise ValueError("base weights must sum to one")
    labels = {i: canonical(pieces[i]) for i in probabilities}
    zero = Fraction(0) if exact else 0.
    output = {i: zero for i in probabilities}

    def visit(ids, prefix, mass, context, used):
        node_mass = sum(probabilities[i] for i in ids)
        ends, children = [], {}
        for i in ids:
            if len(labels[i]) == len(prefix):
                ends.append(i)
            else:
                children.setdefault(labels[i][len(prefix)], []).append(i)
        for i in ends:
            output[i] = mass * (probabilities[i] / node_mass)
        if not children:
            return
        byte_values = sorted(children)
        continuation = sum(probabilities[i] for group in children.values() for i in group)
        conditional = tuple(sum(probabilities[i] for i in children[b]) / continuation for b in byte_values)
        if context not in used and len(byte_values) > 1:
            for layer in range(config.layers):
                conditional = update(conditional, [bits(context, b, layer) for b in byte_values])
        next_used = used | {context}
        for byte, p in zip(byte_values, conditional):
            visit(children[byte], prefix + bytes([byte]),
                  mass * (continuation / node_mass) * p,
                  (context + bytes([byte]))[-config.history:], next_used)

    positive = [i for i, p in probabilities.items() if p > 0]
    visit(positive, b"", total, state.context, state.used)
    out_total = sum(output.values())
    if any(p < 0 for p in output.values()) or (out_total != 1 if exact else abs(out_total - 1) > 1e-10):
        raise ArithmeticError("trie output is not a distribution")
    return output
