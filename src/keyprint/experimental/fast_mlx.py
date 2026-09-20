"""Experimental sparse execution of the pinned MLX reference sampling law.

The frozen engine stays untouched. This advanced pipeline has a distinct
execution identity and is not the default Keyprint.generate path. No process-
wide patches, persistent key cache, calibration or serving acceptance.
"""
import copy
import hashlib
from pathlib import Path

import numpy as np

from .hmac_context import SHAContext
from .batched_tournament import BatchedTokenSourceSession
from .capped_utf8 import CappedPipeline
from ..integrity import verify
from ..sampling import sparse_softmax, identity as sampling_identity
from .._engine.research.keyprint_candidate_v2.adapter import (
    V2Host, sampler, project_supported_logits, ChannelStep, digest,
)
from .._engine.research.keyprint_candidate_v3.adapter import Candidate, V3Host
from .._engine.legacy._impl.research.grouped_canonical_prototype import Profile, pack
from .._engine.legacy._impl.research.token_runtime_profile import RuntimeBoundProfile
from .._engine.legacy._impl.research.token_source_sparse_execution import SparseTokenSourceSession
from .._engine.legacy._impl.research.token_source_policy import TokenSourceSession
from .._engine.legacy._impl.research.token_runtime_host import RuntimeChannelPipeline
from .._engine.legacy._impl.research.byte_trie_source_policy import SourceRequest
from .._engine.legacy._impl.research.token_channel_host import ChannelRequest


class _ContextProfile(RuntimeBoundProfile):
    """Same immutable profile fields, with one response-local HMAC context."""
    def __init__(self, reference):
        if type(reference) is not RuntimeBoundProfile:
            raise TypeError("Optimized execution requires the exact pinned runtime profile")
        for name in ("config", "classes", "tokenizer_identity", "digest", "_domain",
                     "score_namespace_sha256", "original_max_steps"):
            object.__setattr__(self, name, getattr(reference, name))
        object.__setattr__(self, "_state", {})

    def bits(self, key, context, label):
        if type(key) is not bytes or len(key) != 32:
            raise ValueError("key must contain exactly 32 bytes")
        if not label:
            raise ValueError("empty classes have no watermark bits")
        address = (key, context)
        if self._state.get("address") != address:
            context_bytes = pack(context)
            prefix = ((4).to_bytes(4, "big") + len(self._domain).to_bytes(8, "big") + self._domain
                      + len(context_bytes).to_bytes(8, "big") + context_bytes + (4).to_bytes(8, "big"))
            engine = SHAContext(key, prefix, self.config.layers)
            self._state.clear()
            self._state.update(address=address, engine=engine)
        suffix = len(label).to_bytes(8, "big") + label
        engine = self._state["engine"]
        return tuple(engine.digest(layer, suffix)[0] & 1 for layer in range(self.config.layers))


class _ContextSession(BatchedTokenSourceSession):
    def __init__(self, profile, key, **settings):
        super().__init__(_ContextProfile(profile), key, **settings)

    def close(self):
        super().close()
        self.profile._state.clear()


def _upgrade(carrier):
    old = carrier.session
    if type(old) is _ContextSession:
        return
    if (type(old) not in (SparseTokenSourceSession, TokenSourceSession) or old._steps or old._pending is not None
            or old._context or old._used or old._source_output or old._closed):
        raise RuntimeError("Only a fresh sparse carrier may choose optimized execution")
    replacement = _ContextSession(old.profile, old._key, condition=old._condition, request=old._request)
    old.close()
    carrier.session = replacement


class _SparseV2Host(V2Host):
    # Preserve the frozen step's ordering, failures, support checks and receipt
    # format. Only dense softmax execution is replaced with its bitwise oracle-
    # checked sparse implementation. No reference file is modified.
    def step(self, filtered_logits, random_bits):
            self._ready()
            before = len(self._all_ids)
            attempt = {"step": before, "status": "started", "phase": "input",
                       "committed_before": before, "committed_after": before, "random_bit_calls": []}
            self.sampling_attempts.append(attempt)
            try:
                if len(self._all_ids) >= self.binding.profile.config.max_steps:
                    raise RuntimeError("global assistant sample cap reached")
                if not callable(random_bits):
                    raise TypeError("v2 requires an explicit random-bit source, not a float uniform")
                head = project_supported_logits(filtered_logits, model_size=self.model_size,
                                                mapped_size=self.mapped_size)[0]
                attempt["phase"] = "softmax_support_guard"
                base = sparse_softmax(np.asarray(head, dtype=np.float64))
                current = self._current
                attempt["phase"] = "prepare"
                prepared = current.session.prepare(base)
                if not np.array_equal(prepared.probabilities > 0, base > 0):
                    raise ArithmeticError("v2 transformed support differs before commit")
                if not np.array_equal(prepared.probabilities[self._excluded], base[self._excluded]):
                    raise ArithmeticError("excluded control probability changed")
                support = np.flatnonzero(prepared.probabilities > 0)
                positive_weights = tuple(float(v) for v in prepared.probabilities[support])
                integer_weights = sampler.integer_distribution(positive_weights)
                attempt["phase"] = "random_bits"

                def recorded_bits(count):
                    call = {"bit_count": count, "status": "requested"}
                    attempt["random_bit_calls"].append(call)
                    try:
                        value = random_bits(count)
                    except BaseException as exc:
                        call.update(status="raised", exception_type=type(exc).__name__)
                        raise
                    valid = type(value) is int and 0 <= value < (1 << count)
                    call.update(status="returned", returned_type=type(value).__name__,
                                valid_integer=valid,
                                accepted=value < integer_weights.total if valid else None)
                    if type(value) is int:
                        call["value_decimal"] = str(value)
                    elif type(value) in (bool, float, str, type(None)):
                        call["invalid_value_repr"] = repr(value)
                    return value

                sample = sampler.sample_float_weights(positive_weights, recorded_bits)
                token = int(support[sample.token_index])
                attempt.update(phase="commit", selected_token_id=token)
                event = current.session.commit(prepared, token)
                self._all_ids.append(token)
                attempt.update(status="committed", committed_after=len(self._all_ids), phase="render_or_route")
                self.last_sampled_channel = current.name
                self.sampling_records.append({"step": len(self._all_ids)-1, "channel": current.name,
                    "token_id": token, "base_probability_sha256": hashlib.sha256(base.tobytes()).hexdigest(),
                    "prepared_probability_sha256": hashlib.sha256(prepared.probabilities.tobytes()).hexdigest(),
                    "randomness": {"encoding": "exact-integers-as-decimal-strings-v2",
                        "token_index": sample.token_index, "integer_point_decimal": str(sample.integer_point),
                        "total_weight_decimal": str(sample.total_weight),
                        "transcript": [{"bit_count": d.bit_count, "value_decimal": str(d.value),
                                        "accepted": d.accepted} for d in sample.transcript]}})
                if self.binding.token_bytes[token] is None:
                    self._control_ids.append(token)
                    if event is not None:
                        raise ArithmeticError("control produced watermark event")
                    return ChannelStep(token, "", self._route(token))
                emitted = current.append(token, event)
                return ChannelStep(token, emitted if current is self._visible else "", None)
            except BaseException as exc:
                attempt.update(status="failed_after_commit" if len(self._all_ids) > before else "failed_before_commit",
                               committed_after=len(self._all_ids), exception_type=type(exc).__name__)
                if not self._terminal:
                    self._close()
                raise


class _SparseV3Host(V3Host, _SparseV2Host):
    def __init__(self, *args, **settings):
        super().__init__(*args, **settings)
        _upgrade(self._visible)

    def _route(self, token):
        # Reuse frozen channel policy, then upgrade only newly created carriers.
        result = RuntimeChannelPipeline._route(self, token)
        if not self._terminal:
            _upgrade(self._current)
        return result


def execution_specification():
    """Bind every experimental implementation and reporting source."""
    return {
            "sources": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                        for name in ("fast_mlx.py", "fast_caller.py", "fast_public.py", "fast_reporting.py", "hmac_context.py", "batched_tournament.py", "capped_utf8.py")},
            "sampling": sampling_identity(), "request_local_hmac_context": True,
            "reference_results_transfer": False}


class FastCandidate:
    """Reusable bound candidate; response state remains in fresh pipelines."""
    def __init__(self, reference):
        verify()
        if type(reference) is not Candidate:
            raise TypeError("An exact frozen candidate is required")
        self.reference = reference
        identity = copy.deepcopy(reference.identity)
        spec = identity["specification"]
        spec["version"] = "keyprint-mlx-sparse-experimental-v1"
        spec["reference_runtime_sha256"] = identity["runtime_profile_sha256"]
        spec["execution"] = execution_specification()
        identity.update(version=spec["version"], runtime_profile_sha256=digest(spec))
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
        result["score_identity_scope"] = "Unchanged reference score namespace; experimental execution does not transfer acceptance"
        return result

    def pipeline(self, key, *, condition, purpose="general", source_text=None,
                 allow_thinking=False, allow_tools=False):
        self.reference._base._scorer.check_key(key)
        raw = _SparseV3Host(key, condition=condition, binding=self.reference._base._binding,
                           request=SourceRequest(purpose, source_text),
                           channels=ChannelRequest(allow_thinking, allow_tools),
                           v2_identity=self.identity, filter_settings=self.filter_settings)
        return CappedPipeline(raw, hashlib.sha256(key).hexdigest())


def pipeline(key, *, condition="marked", temperature=.7, top_k=100, max_steps=2048,
             purpose="general", source_text=None, allow_thinking=False, allow_tools=False):
    """One explicit advanced supplied-head pipeline; close after every response."""
    reference = Candidate(temperature=temperature, top_k=top_k, max_steps=max_steps)
    return FastCandidate(reference).pipeline(key, condition=condition, purpose=purpose,
                                            source_text=source_text, allow_thinking=allow_thinking,
                                            allow_tools=allow_tools)
