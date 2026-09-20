"""Reconcile a completed, non-streaming CPU pilot response with its token bytes.

Call this after the native framework returns. This does not alter sampling or
replace the separate check that returned IDs match the selection journal.
"""
from dataclasses import dataclass

from ..backends.bytelevel import ByteLevelBinding


@dataclass(frozen=True)
class NativeCompletion:
    text: str
    host_text: str
    token_ids: tuple[int, ...]
    completion: str
    pending_utf8: bytes

    @property
    def carrier_rendering(self) -> dict[str, object]:
        return {"pending_utf8_hex": self.pending_utf8.hex(),
                "host_text_adjusted": self.text != self.host_text,
                "policy": "strict-bytes-with-pending-suffix-at-token-cap-v1"}


def finalize_completion(binding: ByteLevelBinding, *, token_ids: list[int] | tuple[int, ...],
                        text: str, finish_reason: str, max_tokens: int) -> NativeCompletion:
    """Validate host completion and return text without invented replacement bytes.

    At an exact length cap, an unfinished UTF-8 suffix is retained separately.
    A host may omit that suffix or render it as one U+FFFD; only that known tail
    is normalized. Other text differences, malformed bytes and unknown finish
    reasons fail. EOS must be terminal and decode strictly, including at a cap.
    Empty visible text is possible when the first token is a character fragment.
    """
    if type(max_tokens) is not int or not 1 <= max_tokens <= 1024:
        raise ValueError("native completion requires a token cap from 1 to 1024")
    if (not isinstance(token_ids, (list, tuple)) or not 0 < len(token_ids) <= max_tokens
            or any(type(i) is not int or not 0 <= i < len(binding.pieces) for i in token_ids)):
        raise ValueError("native completion requires valid returned token IDs within the cap")
    if not isinstance(text, str) or finish_reason not in ("stop", "length"):
        raise ValueError("native completion requires text and a stop or length finish reason")
    ids = tuple(token_ids)
    eos = ids[-1] in binding.eos_ids
    ordinary = ids[:-1] if eos else ids
    if any(binding.pieces[i] is None or i in binding.eos_ids for i in ordinary):
        raise ValueError("native completion permits only ordinary tokens followed by terminal EOS")
    if finish_reason == "stop" and not eos:
        raise ValueError("native stop completion must include terminal EOS")
    if finish_reason == "length" and len(ids) != max_tokens:
        raise ValueError("native length completion must reach the exact token cap")
    if eos:
        visible, pending = binding.render(list(ids)), b""
    else:
        visible, pending = binding.render_at_limit(list(ids), max_tokens)
    if text != visible and not (pending and text == visible + "\ufffd"):
        raise ValueError("host text differs from returned token bytes")
    return NativeCompletion(visible, text, ids, "eos" if finish_reason == "stop" else "length", pending)
