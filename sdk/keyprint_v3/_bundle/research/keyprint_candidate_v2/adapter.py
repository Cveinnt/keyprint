"""Versioned future candidate: inherited channel policy, exact integer sampler.

Only supplied logits and explicit independent random bits are accepted. No model,
key discovery, fit, or frozen-file modifications occur. Empirical v1 receipts do
not apply to this runtime. Arbitrary finite-logit softmax underflow is rejected.
"""
from dataclasses import asdict, replace
import copy
import hashlib
import json
from pathlib import Path
import sys
from threading import get_ident

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
INSTALLED = ROOT / "release/keyprint-0.0.3rc1/installed-audit"
sys.path.insert(0, str(INSTALLED))
import keyprint as inherited_sdk
if Path(inherited_sdk.__file__).resolve() != INSTALLED / "keyprint/__init__.py":
    raise RuntimeError("v2 adapter requires the exact local installed-audit SDK")
from keyprint._impl.research.byte_trie_source_policy import SourceRequest
from keyprint._impl.research.token_channel_host import ChannelRequest, ChannelStep
from keyprint._impl.research.grouped_model_support_boundary import project_supported_logits
from keyprint._impl.research.token_runtime_sparse_host import SparseRuntimeChannelPipeline
import keyprint_exact_categorical_v2 as sampler


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def version_identity(base):
    files = [*sorted((INSTALLED / "keyprint").rglob("*.py")),
             *sorted((INSTALLED / "keyprint").rglob("*.json")),
             Path(sampler.__file__), Path(__file__), Path(__file__).with_name("__init__.py")]
    reporting = ROOT / "research/keyprint_reporting_v2.py"
    if reporting.exists():
        files.append(reporting)
    dependencies = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    spec = {"version": "keyprint-candidate-v2-proposed-2026-09-09",
            "inherited_engine_identity": base.identity,
            "randomness_api": "independent_uniform_random_bits(k)_integer_v2",
            "sampling": "exact_integer_ratios_of_prepared_binary64_weights_with_rejection",
            "softmax": "math_exp_fsum_binary64_reject_finite_support_loss_before_prepare",
            "random_rejection_cap": 1024, "dependencies": dependencies}
    return {"version": spec["version"], "runtime_profile_sha256": digest(spec),
            "score_namespace_sha256": base.identity["score_namespace_sha256"],
            "inherited_engine_identity": dict(base.identity),
            "max_steps": base.identity["max_steps"], "deployment_calibrated": False,
            "specification": spec}


class V2Host(SparseRuntimeChannelPipeline):
    def __init__(self, *args, v2_identity, **kwargs):
        super().__init__(*args, **kwargs)
        self.runtime_identity = copy.deepcopy(v2_identity)
        self.channel_profile_sha256 = digest({"v2_runtime": v2_identity["runtime_profile_sha256"],
                                               "inherited_channel_profile": self.channel_profile_sha256})
        self.host_route_sha256 = digest({"v2_channel": self.channel_profile_sha256,
                                        "inherited_host_route": self.host_route_sha256})
        self.sampling_records = []
        self.sampling_attempts = []

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
            base = np.asarray(sampler.supported_softmax(tuple(float(v) for v in head)), dtype=np.float64)
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

    def finish(self):
        value = super().finish()
        runtime = self.runtime_identity["runtime_profile_sha256"]
        carrier = lambda result: replace(result, profile_sha256=runtime) if result is not None else None
        return replace(value, visible=carrier(value.visible), reasoning=carrier(value.reasoning),
                       tools=tuple(carrier(v) for v in value.tools), runtime_profile_sha256=runtime)


class Candidate:
    """Local future candidate adapter. This is not a released or validated SDK."""
    def __init__(self, *, max_steps=2048):
        self._base = inherited_sdk.Candidate(max_steps=max_steps)
        self._identity = version_identity(self._base)
        if self.identity["runtime_profile_sha256"] == self._base.identity["runtime_profile_sha256"]:
            raise AssertionError("v2 must have a distinct runtime identity")

    @property
    def identity(self):
        return copy.deepcopy(self._identity)

    def pipeline(self, key, *, condition, purpose="general", source_text=None,
                 allow_thinking=False, allow_tools=False):
        self._base._scorer.check_key(key)
        raw = V2Host(key, condition=condition, binding=self._base._binding,
                     request=SourceRequest(purpose, source_text),
                     channels=ChannelRequest(allow_thinking, allow_tools), v2_identity=self.identity)
        return Pipeline(raw, hashlib.sha256(key).hexdigest())

    def score_literal(self, text, key):
        from keyprint_reporting_v2 import score_literal_report
        report = score_literal_report(self._base, text, key)
        return {**report, "candidate_identity": self.identity,
                "score_identity_scope": "Inherited score law; new sampling runtime; no empirical v1 calibration transfers."}


class Pipeline(inherited_sdk.Pipeline):
    def step(self, filtered_logits, random_bits):
        value = self._raw.step(filtered_logits, random_bits)
        if value.stopped is not None:
            self._result = value.stopped
        return value

    def receipt(self):
        if get_ident() != self._raw._owner:
            raise RuntimeError("pipeline belongs to another thread")
        return copy.deepcopy({"runtime": self._raw.runtime_identity,
                "execution_identity_sha256": self._raw.runtime_identity["runtime_profile_sha256"],
                "key_commitment": self._key_commitment, "condition": self._raw.condition,
                "committed_token_ids": list(self.committed_token_ids),
                "channel_profile_sha256": self._raw.channel_profile_sha256,
                "host_route_sha256": self._raw.host_route_sha256,
                "sampling_records": list(self._raw.sampling_records),
                "sampling_attempts": list(self._raw.sampling_attempts),
                "attempted_bit_requests": sum(len(a["random_bit_calls"]) for a in self._raw.sampling_attempts),
                "completed_bit_returns": sum(c["status"] == "returned" for a in self._raw.sampling_attempts for c in a["random_bit_calls"]),
                "closed": self._raw._terminal, "finalized": self._result is not None,
                "final": asdict(self._result) if self._result else None, "calibrated": False})
