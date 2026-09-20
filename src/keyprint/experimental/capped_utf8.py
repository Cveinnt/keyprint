"""Lossless receipt for a token cap reached inside a UTF-8 character.

Only an explicit length limit may omit an unfinished character from rendered
text. Sampled tokens and pending bytes remain in the receipt. EOS, controls and
ordinary finish retain the frozen strict decoder behavior.
"""
import codecs
from dataclasses import dataclass

from .._engine.legacy._impl.research.token_source_host import TokenFinished
from .._engine.legacy._impl.research.grouped_null_score import score
from .._engine.research.keyprint_candidate_v3.adapter import Pipeline


@dataclass(frozen=True)
class CappedTokenFinished(TokenFinished):
    rendering_status: str = "incomplete_utf8_at_token_limit"
    pending_utf8_hex: str = ""


def retain_capped_carrier(carrier):
    """Finalize a valid byte prefix without inventing or dropping sampled bytes."""
    pending = carrier.decoder.getstate()[0]
    if not pending:
        return
    if carrier.finished is not None or carrier.closed:
        raise RuntimeError("Only an unfinished carrier may stop inside a character")
    sampled = b"".join(carrier.binding.token_bytes[i] for i in carrier.ids)
    decoder = codecs.getincrementaldecoder("utf-8")("strict")
    if (decoder.decode(sampled, final=False) != carrier.text
            or decoder.getstate()[0] != pending
            or carrier.text.encode("utf-8") + pending != sampled):
        raise ValueError("Capped carrier bytes do not match rendered text and pending suffix")
    carrier.finished = CappedTokenFinished(
        text=carrier.text, token_ids=tuple(carrier.ids),
        profile_sha256=carrier.binding.profile.digest, events_match=False,
        unavailable_reason="Token limit split a UTF-8 character; rendered prefix is not the complete sampled carrier",
        generation_score=score(carrier.events, carrier.binding.profile.config.layers),
        text_score=None, replay_note="All committed token IDs and pending UTF-8 bytes are retained",
        pending_utf8_hex=pending.hex(),
    )


class CappedPipeline(Pipeline):
    def finish_at_limit(self, max_tokens):
        """Finish only at the caller's exact declared token cap; never sample more."""
        self._raw._ready()
        if (type(max_tokens) is not int or max_tokens < 0
                or len(self.committed_token_ids) != max_tokens or self._raw._selected_eos):
            raise ValueError("Length finalization requires the exact reached token cap")
        try:
            retain_capped_carrier(self._raw._current)
            return self.finish()
        finally:
            self.close()
