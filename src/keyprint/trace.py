"""Exact committed-token rendering for optional, inspectable generation demos."""
from __future__ import annotations

import codecs
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class TokenChoice:
    index: int
    token_id: int
    bytes_hex: str | None
    text: str
    start: int
    end: int
    kind: str


def token_trace(report: dict, text: str, pieces: Sequence[bytes | None]) -> tuple[TokenChoice, ...]:
    """Project only committed IDs and public token bytes, never keys/randomness.

    UTF-8 pieces may emit no text until a later token completes a character.
    Offsets count Unicode code points, not UTF-16 code units. Exact reconstruction
    is required; a token-capped unfinished suffix remains in bytes_hex.
    """
    payload = report.get('payload', report)
    ids = payload['committed_token_ids']
    if not isinstance(ids, list) or len(ids) > 1024:
        raise ValueError('Invalid committed-token trace')
    decoder = codecs.getincrementaldecoder('utf-8')('strict')
    rendered, result = '', []
    for index, token_id in enumerate(ids):
        if type(token_id) is not int or not 0 <= token_id < len(pieces):
            raise ValueError('Trace token outside declared binding')
        piece = pieces[token_id]
        start = len(rendered)
        if piece is None:
            emitted, kind = '', 'control'
        else:
            emitted = decoder.decode(piece, final=False)
            kind = 'text' if emitted else 'pending_bytes'
        rendered += emitted
        result.append(TokenChoice(index, token_id, None if piece is None else piece.hex(),
                                  emitted, start, len(rendered), kind))
    if rendered != text:
        raise ValueError('Committed tokens do not reconstruct the visible response')
    pending = decoder.getstate()[0]
    if pending and payload.get('completion') != 'length':
        raise ValueError('Incomplete UTF-8 outside token-limit completion')
    return tuple(result)
