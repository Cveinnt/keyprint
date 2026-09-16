from dataclasses import dataclass
from .grouped_null_score import NullScore

@dataclass(frozen=True)
class TokenFinished:
    text: str
    token_ids: tuple[int, ...]
    profile_sha256: str
    events_match: bool
    unavailable_reason: str | None
    generation_score: NullScore
    text_score: NullScore | None
    replay_note: str | None
    deployment_calibrated: bool = False
