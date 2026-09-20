"""Explicit tokenizer binding for the experimental portable text profile."""
from __future__ import annotations

from dataclasses import dataclass
import codecs
import hashlib
import json


@dataclass(frozen=True)
class ByteLevelBinding:
    pieces: tuple[bytes | None, ...]
    eos_ids: frozenset[int]
    digest: str

    @classmethod
    def create(cls, serialized: str, *, vocabulary_size: int,
               special_ids: list[int], eos_ids: list[int]) -> ByteLevelBinding:
        data = json.loads(serialized)
        if data.get("model", {}).get("type") != "BPE" or data.get("decoder", {}).get("type") != "ByteLevel":
            raise ValueError("portable profile requires a BPE tokenizer with a ByteLevel decoder")
        if data.get("normalizer") is not None:
            raise ValueError("portable profile does not support tokenizer normalization")
        if type(vocabulary_size) is not int or not 1 <= vocabulary_size <= 151669:
            raise ValueError("portable vocabulary must contain 1 to 151669 tokens")
        if not eos_ids or any(type(i) is not int or not 0 <= i < vocabulary_size for i in [*special_ids, *eos_ids]):
            raise ValueError("valid, explicit special and EOS token IDs required")
        # GPT-2 ByteLevel alphabet, reconstructed bijectively without lossy
        # single-token Unicode decoding (tokens can contain UTF-8 fragments).
        base = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
        alphabet = {chr(i): i for i in base}
        for offset, byte in enumerate(i for i in range(256) if i not in base):
            alphabet[chr(256 + offset)] = byte
        vocab = dict(data["model"]["vocab"])
        special = set(special_ids)
        added_ids: set[int] = set()
        for added in data.get("added_tokens", []):
            index, content = added["id"], added["content"]
            if type(index) is not int or not 0 <= index < vocabulary_size or index in added_ids:
                raise ValueError("added tokens require unique IDs within the model head")
            added_ids.add(index)
            if content in vocab and vocab[content] != index:
                raise ValueError("added token conflicts with the base vocabulary")
            if not added.get("special", False):
                # These literals have identical UTF-8 and ByteLevel bytes.
                # Reject whitespace/Unicode and matching transformations rather
                # than guessing how an added token affects adjacent bytes.
                if (index in special or not content
                        or any(not 33 <= ord(c) <= 126 for c in content)
                        or any(added.get(flag, False) for flag in
                               ("single_word", "lstrip", "rstrip", "normalized"))):
                    raise ValueError("ordinary added tokens require nonempty ASCII literals without matching transformations")
            else:
                special.add(index)
            vocab[content] = index
        if not set(eos_ids) <= special:
            raise ValueError("EOS tokens must be declared special tokens")
        if any(type(i) is not int or not 0 <= i < vocabulary_size for i in vocab.values()):
            raise ValueError("tokenizer vocabulary exceeds the model head")
        if len(set(vocab.values())) != len(vocab):
            raise ValueError("duplicate token IDs in tokenizer vocabulary")
        pieces: list[bytes | None] = [None] * vocabulary_size
        for token, index in vocab.items():
            if index not in special:
                try:
                    pieces[index] = bytes(alphabet[c] for c in token)
                except KeyError as exc:
                    raise ValueError("token is outside the declared ByteLevel alphabet") from exc
                if not pieces[index]:
                    raise ValueError("empty ordinary token is unsupported")
        identity = json.dumps({"profile": "keyprint-portable-bytelevel-v1-experimental",
                               "tokenizer": data, "vocabulary_size": vocabulary_size,
                               "special_ids": sorted(special), "eos_ids": sorted(set(eos_ids)),
                               "channels": "visible_text_only", "source_policy": "general"},
                              sort_keys=True, separators=(",", ":"))
        return cls(tuple(pieces), frozenset(eos_ids), hashlib.sha256(identity.encode()).hexdigest())

    def render(self, ids: list[int]) -> str:
        return b"".join(self.pieces[i] or b"" for i in ids).decode("utf-8", errors="strict")

    def render_at_limit(self, ids: list[int], max_tokens: int) -> tuple[str, bytes]:
        """Return a valid prefix and retained suffix only at an exact token cap.

        This never decodes with replacement or consumes an additional token.
        EOS and malformed UTF-8 remain strict errors through ordinary render().
        """
        if (type(max_tokens) is not int or max_tokens < 1 or len(ids) != max_tokens
                or any(type(i) is not int or not 0 <= i < len(self.pieces)
                       or self.pieces[i] is None or i in self.eos_ids for i in ids)):
            raise ValueError("Length rendering requires the exact token cap and ordinary token IDs")
        sampled = b"".join(self.pieces[i] for i in ids)
        decoder = codecs.getincrementaldecoder("utf-8")("strict")
        text = decoder.decode(sampled, final=False)
        pending = decoder.getstate()[0]
        if text.encode("utf-8") + pending != sampled:
            raise ValueError("Rendered text and pending suffix differ from sampled bytes")
        return text, pending
