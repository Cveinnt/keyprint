"""Response-local SHA-256 HMAC templates for the fixed 32-byte SDK key.

Uses standard inner/outer pads and hashlib's SHA-256 implementation. No PRF
address, digest truncation, key format or score law changes. The caller owns and
clears each context; no module-level keyed state or alternate crypto fallback.
"""
import hashlib


class SHAContext:
    def __init__(self, key, prefix, layers):
        if type(key) is not bytes or len(key) != 32:
            raise ValueError("exactly 32 key bytes required")
        padded = key + bytes(32)
        self.outer = hashlib.sha256(bytes(v ^ 0x5c for v in padded))
        base = hashlib.sha256(bytes(v ^ 0x36 for v in padded))
        base.update(prefix)
        self.templates = []
        for layer in range(layers):
            inner = base.copy()
            inner.update(layer.to_bytes(4, "big"))
            self.templates.append(inner)

    def digest(self, layer, suffix):
        inner = self.templates[layer].copy()
        inner.update(suffix)
        outer = self.outer.copy()
        outer.update(inner.digest())
        return outer.digest()
