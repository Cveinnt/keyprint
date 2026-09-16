"""Pinned real-tokenizer binding for the research-only grouped prototype.

Reject added/control token IDs and non-exact text round trips. No normalization
of caller-visible text, tag stripping, model invocation or detection decision.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Sequence
from tokenizers import Tokenizer
from .grouped_canonical_prototype import Config, Event, Profile, replay_events
from .check_canonical_event_alignment import byte_alphabet

ROOT = Path(__file__).resolve().parents[1]
TOKENIZER_PATH = ROOT / "release/tokenizer/qwen3-8b-4bit/tokenizer.json"
TOKENIZER_SHA = "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"
PROTOTYPE_SHA = "d74462e9b6e6d47fd94bfb41e99bfea993a7adf7a3181a692bfa0ab54116fa2c"


@dataclass(frozen=True)
class Replay:
    text_sha256: str
    tokenizer_sha256: str
    profile_sha256: str
    token_ids: tuple[int, ...]
    events: tuple[Event, ...]
    calibrated: bool = False


class Binding:
    def __init__(self, config=Config()):
        raw = TOKENIZER_PATH.read_bytes()
        if hashlib.sha256(raw).hexdigest() != TOKENIZER_SHA:
            raise ValueError("tokenizer hash mismatch")
        prototype = Path(__file__).with_name("grouped_canonical_prototype.py")
        if hashlib.sha256(prototype.read_bytes()).hexdigest() != PROTOTYPE_SHA:
            raise ValueError("frozen prototype hash mismatch")
        config_data = json.loads(raw)
        if config_data["model"]["type"] != "BPE" or config_data["decoder"]["type"] != "ByteLevel":
            raise ValueError("unsupported tokenizer implementation")
        vocab = config_data["model"]["vocab"]
        added = config_data["added_tokens"]
        excluded = {row["id"] for row in added}
        # Deliberately exclude all added/control markers, including special=False
        # entries such as <think>. This is a local visible-text policy, not Claude's.
        all_ids = set(vocab.values()) | excluded
        if all_ids != set(range(max(all_ids) + 1)):
            raise ValueError("tokenizer IDs are not dense; padding policy required")
        mapping = [None] * len(all_ids)
        alphabet = byte_alphabet()
        for symbol, idx in vocab.items():
            if idx not in excluded:
                mapping[idx] = bytes(alphabet[c] for c in symbol)
        self._tokenizer = Tokenizer.from_str(raw.decode())
        for symbol, byte in alphabet.items():
            if self._tokenizer.decode([vocab[symbol]], skip_special_tokens=False) != bytes([byte]).decode("utf-8", "replace"):
                raise ValueError("byte alphabet does not match tokenizer decoder")
        self._pieces = tuple(mapping)
        self._excluded = frozenset(excluded)
        self._profile = Profile(self._pieces,
                                tokenizer_identity=f"qwen3-8b:{TOKENIZER_SHA}:visible-control-reject-v1",
                                config=config)

    @property
    def profile(self):
        return self._profile

    @property
    def token_bytes(self):
        return self._pieces

    def decode_visible(self, ids: Sequence[int]) -> str:
        if len(ids) > self.profile.config.max_steps:
            raise ValueError("visible path exceeds response cap")
        for idx in ids:
            self.profile.label(idx)
            if idx in self._excluded:
                raise ValueError("added/control tokens are unavailable on visible-text route")
        text = b"".join(self._pieces[idx] for idx in ids).decode("utf-8", "strict")
        if self._tokenizer.decode(list(ids), skip_special_tokens=False) != text:
            raise ValueError("raw bytes disagree with reference decoder")
        return text

    def encode_visible(self, text: str) -> tuple[int, ...]:
        if not isinstance(text, str):
            raise TypeError("visible input must be a string")
        if len(text.encode("utf-8", "strict")) > 2 * 1024 * 1024:
            raise ValueError("visible input exceeds byte cap")
        ids = tuple(self._tokenizer.encode(text, add_special_tokens=False).ids)
        if self.decode_visible(ids) != text:
            raise ValueError("tokenizer normalizes visible input; exact replay unavailable")
        return ids

    def replay_text(self, text: str, key: bytes) -> Replay:
        ids = self.encode_visible(text)
        return Replay(hashlib.sha256(text.encode()).hexdigest(), TOKENIZER_SHA,
                      self.profile.digest, ids, replay_events(self.profile, key, ids))
