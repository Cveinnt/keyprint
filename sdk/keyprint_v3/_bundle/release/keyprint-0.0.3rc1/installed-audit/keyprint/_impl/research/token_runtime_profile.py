"""Design-level runtime bound extension; not a new calibrated host or experiment.

The full profile digest commits the larger cap. Keyed bits intentionally remain
in the frozen score namespace. Consumers must report both identities; a scorer
that derives HMAC addresses from the new full digest is incompatible.
"""
from dataclasses import dataclass, replace
import hashlib
import json

from .grouped_canonical_prototype import Profile


@dataclass(frozen=True, init=False)
class RuntimeBoundProfile(Profile):
    score_namespace_sha256: str
    original_max_steps: int

    def __init__(self, base: Profile, *, max_steps: int):
        if type(base) is not Profile:
            raise TypeError("extend an original Profile exactly once")
        config = replace(base.config, max_steps=max_steps)
        if max_steps <= base.config.max_steps:
            raise ValueError("runtime extension requires a strictly larger cap")
        spec = {"version": "token-runtime-bound-v1", "score_namespace_sha256": base.digest,
                "original_max_steps": base.config.max_steps, "max_steps": max_steps,
                "context_history": "continuous_no_reset_no_eviction",
                "keyed_addresses": "unchanged_original_profile_domain"}
        digest = hashlib.sha256(json.dumps(spec, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        for name, value in {"config": config, "classes": base.classes,
                            "tokenizer_identity": base.tokenizer_identity,
                            "digest": digest, "_domain": base._domain,
                            "score_namespace_sha256": base.digest,
                            "original_max_steps": base.config.max_steps}.items():
            object.__setattr__(self, name, value)

    def identity_receipt(self):
        return {"runtime_profile_sha256": self.digest,
                "score_namespace_sha256": self.score_namespace_sha256,
                "original_max_steps": self.original_max_steps,
                "max_steps": self.config.max_steps,
                "deployment_calibrated": False}
