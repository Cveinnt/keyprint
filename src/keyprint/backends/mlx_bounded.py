"""Frozen reference sampling with separately identified, lossless cap finalization.

The archived engine is unchanged. Only exact length completion may retain an
unfinished UTF-8 suffix; EOS and ordinary pipeline.finish() remain strict.
"""
import copy
import hashlib
from pathlib import Path

from ..integrity import verify
from .._engine.research.keyprint_candidate_v3.adapter import (
    Candidate, V3Host, SourceRequest, ChannelRequest, digest,
)
from ..experimental.capped_utf8 import CappedPipeline
from ..experimental.fast_public import FastPublicCandidate

VERSION = "keyprint-mlx-bounded-reference-v1"


def execution_specification():
    shared = Path(__file__).parents[1] / "experimental"
    paths = [Path(__file__), *(shared / name for name in (
        "fast_caller.py", "fast_public.py", "fast_reporting.py", "capped_utf8.py"))]
    return {"sources": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
            "sampling": "unchanged_frozen_v3_host",
            "length_finalization": "valid_utf8_prefix_with_all_tokens_and_pending_bytes",
            "reference_results_transfer": False}


class BoundedReferenceCandidate:
    """Use the frozen sampling host with an independently bound finalization."""
    def __init__(self, reference):
        verify()
        if type(reference) is not Candidate:
            raise TypeError("An exact frozen candidate is required")
        self.reference = reference
        identity = copy.deepcopy(reference.identity)
        spec = identity["specification"]
        spec["version"] = VERSION
        spec["reference_runtime_sha256"] = identity["runtime_profile_sha256"]
        spec["execution"] = execution_specification()
        identity.update(version=VERSION, runtime_profile_sha256=digest(spec))
        self._identity = identity

    @property
    def identity(self):
        return copy.deepcopy(self._identity)

    @property
    def filter_settings(self):
        return self.reference.filter_settings

    def score_literal(self, text, key):
        result = self.reference.score_literal(text, key)
        result["candidate_identity"] = self.identity
        result["score_identity_scope"] = "Unchanged reference score law; bounded finalization does not transfer empirical acceptance"
        return result

    def pipeline(self, key, *, condition, purpose="general", source_text=None,
                 allow_thinking=False, allow_tools=False):
        self.reference._base._scorer.check_key(key)
        raw = V3Host(key, condition=condition, binding=self.reference._base._binding,
                     request=SourceRequest(purpose, source_text),
                     channels=ChannelRequest(allow_thinking, allow_tools),
                     v2_identity=self.identity, filter_settings=self.filter_settings)
        return CappedPipeline(raw, hashlib.sha256(key).hexdigest())


class BoundedReferencePublicCandidate(FastPublicCandidate):
    core_type = BoundedReferenceCandidate
    facade_version = "keyprint-bounded-reference-facade-v1"
    integration_status = "bounded_reference_lifecycle_qualification_pending"
    package_scope = "Reference sampling with bounded UTF-8 finalization; no new scientific or serving acceptance"
