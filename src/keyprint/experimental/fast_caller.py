"""Shared caller for separately bound capped-reference and sparse execution.

Preserves its durable draw/commit ordering, caps, failure receipts and cache
lifecycle. The frozen caller remains untouched. Exact candidate classes bind
their sampling host and finalization; shared frozen journal/failure types keep
close failures in the same consumed-work accounting.
"""
from __future__ import annotations
from dataclasses import asdict
import hashlib
from pathlib import Path
from typing import Any
import numpy as np
from .._engine.research.keyprint_candidate_v3_caller import (
    DurableJournal, ResponseFailure, PUBLIC_FIXTURE_KEY, V1_RUNTIME, digest,
)


def run_response(candidate, model, prompt_ids, *, key, condition, random_bits,
                 journal, reserve, max_tokens=1024, purpose="general", source_text=None,
                 allow_thinking=True, allow_tools=False, temperature=None, top_k=None,
                 prefill_step_size=2048, max_random_draws=16384,
                 max_random_bits=4 * 1024 * 1024, max_model_calls=9216,
                 capture_public_fixture=False, backend=None, cache_factory=None,
                 constraint=None):
    """One response; bit source and durable journal precede every sampled commit.

    Inject ``backend``/``cache_factory`` for supplied fake-model tests. Production
    defaults lazily import MLX/cache helpers, but never construct or load a model.
    Errors do not return a negative detector result or permit continuing a closed
    pipeline. A failed attempt is consumed and its journal must be retained.
    The returned ``response`` contains the final generated carriers for the caller.
    It is deliberately added only after journaling the sanitized terminal event.
    """
    prompt = tuple(prompt_ids)
    if not prompt or len(prompt) > 8192 or any(type(i) is not int or not 0 <= i < 151669 for i in prompt):
        raise ValueError("bounded nonempty mapped prompt required")
    if type(max_tokens) is not int or not 0 <= max_tokens <= 1024:
        raise ValueError("response budget must be between zero and 1024")
    if type(prefill_step_size) is not int or prefill_step_size <= 0:
        raise ValueError("positive prefill step size required")
    for value, cap, name in ((max_random_draws, 1048576, "random draw"),
                             (max_random_bits, 16777216, "random bit"),
                             (max_model_calls, 9216, "model call")):
        if type(value) is not int or not 0 <= value <= cap:
            raise ValueError(f"invalid {name} cap")
    if type(key) is not bytes or len(key) != 32:
        raise ValueError("key must contain exactly 32 bytes")
    if type(capture_public_fixture) is not bool or (capture_public_fixture and key != PUBLIC_FIXTURE_KEY):
        raise ValueError("content capture requires the explicit public fixture key")
    if any(not callable(fn) for fn in (model, random_bits, reserve)):
        raise TypeError("model, random_bits and reserve must be callable")
    if not isinstance(journal, DurableJournal) or journal.broken or journal.sequence != 0:
        raise ValueError("a fresh durable journal is required; attempts cannot resume")
    if constraint is not None:
        from ..structured import JsonConstraint
        if type(constraint) is not JsonConstraint or constraint.size != 151669:
            raise ValueError("an exact pinned JSON constraint is required")
        if allow_thinking or allow_tools or purpose != "general" or source_text is not None:
            raise ValueError("JSON constraints require the visible general route")
    from .fast_mlx import FastCandidate
    expected_version = "keyprint-mlx-sparse-experimental-v1"
    if type(candidate) is not FastCandidate:
        from ..backends.mlx_bounded import BoundedReferenceCandidate, VERSION
        from .native_mlx import NativeCandidate, VERSION as NATIVE_VERSION
        if type(candidate) is NativeCandidate:
            expected_version = NATIVE_VERSION
        elif type(candidate) is BoundedReferenceCandidate:
            expected_version = VERSION
        else:
            raise ValueError("caller requires an exact bound Keyprint candidate class")
    settings = candidate.filter_settings
    if temperature is not None and temperature != settings['temperature']:
        raise ValueError("caller temperature must match the candidate-bound shared filter")
    if top_k is not None and top_k != settings['top_k']:
        raise ValueError("caller top-k must match the candidate-bound shared filter")
    temperature, top_k = settings['temperature'], settings['top_k']
    identity = candidate.identity
    runtime = identity.get("runtime_profile_sha256")
    namespace = identity.get("score_namespace_sha256")
    specification = identity.get("specification")
    if (identity.get("version") != expected_version
            or not isinstance(runtime, str) or len(runtime) != 64 or runtime == V1_RUNTIME
            or type(specification) is not dict or digest(specification) != runtime
            or specification.get("randomness_api") != "independent_uniform_random_bits(k)_integer_v2"
            or namespace != specification.get("inherited_engine_identity", {}).get("score_namespace_sha256")
            or identity.get("deployment_calibrated") is not False):
        raise ValueError("caller requires the bound candidate class, runtime and score namespace")
    bound_filter = specification['shared_filter']['specification']
    if settings != {name: bound_filter[name] for name in ('temperature','top_k','max_logit_gap')}:
        raise ValueError("candidate filter settings differ from its runtime identity")
    pipeline = candidate.pipeline(key, condition=condition, purpose=purpose, source_text=source_text,
        allow_thinking=allow_thinking, allow_tools=allow_tools)
    cache = None
    state: dict[str, Any] = {"phase": "start", "model_calls": 0, "prefill_calls": 0,
                            "bit_requests": 0, "bit_values_obtained": 0,
                            "bit_values_journaled": 0, "bits_requested": 0,
                            "sample_attempts": 0, "sampled_tokens": 0}
    structured = ({"identity": constraint.identity, "status": "incomplete", "schema_validated": False}
                  if constraint is not None else None)

    def record(event):
        journal.append(event)

    def snapshot(outcome, error_type=None):
        state["sampled_tokens"] = len(pipeline.committed_token_ids)
        result = {"kind": "response_terminal", "outcome": outcome, **state,
                  "journal_broken": journal.broken, "error_type": error_type,
                  "runtime_profile_sha256": runtime, "score_namespace_sha256": namespace,
                  "caller_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  "verdict": None, "deployment_calibrated": False,
                  "empirical_v1_results_transfer": False, "empirical_v2_results_transfer": False}
        if capture_public_fixture:
            result["public_fixture_sdk"] = pipeline.receipt()
        if structured is not None:
            result["structured_output"] = structured.copy()
        return result

    def finish_constraint(response, completion):
        if constraint is not None:
            state["phase"] = "structured_validation"
            structured.update(constraint.finish(response["visible"]["text"], completion))

    def forward(ids, *, prefill):
        state["phase"] = "model_forward_reservation"
        if state["model_calls"] >= max_model_calls:
            raise RuntimeError("model call cap exhausted")
        metadata = {"index": state["model_calls"], "prefill": prefill, "input_length": len(ids)}
        reserve("model_forward", metadata)
        event = {"kind": "model_forward_reserved", **metadata}
        if capture_public_fixture:
            event["public_fixture_input_ids"] = ids.tolist()
        record(event)
        state["model_calls"] += 1
        state["prefill_calls"] += int(prefill)
        state["phase"] = "model_forward"
        output = model(ids[None], cache=cache)
        if output.ndim != 3 or output.shape != (1, len(ids), pipeline.model_vocabulary_size):
            raise ValueError("model output must match batch, input and bound vocabulary")
        return output

    def bits(count):
        state["phase"] = "random_bits_reservation"
        if type(count) is not int or count <= 0:
            raise ValueError("positive random bit count required")
        if state["bit_requests"] >= max_random_draws or state["bits_requested"] + count > max_random_bits:
            raise RuntimeError("random bit resource cap exhausted")
        metadata = {"index": state["bit_requests"], "sample_index": state["sample_attempts"] - 1, "bits": count}
        reserve("random_bits", metadata)
        record({"kind": "random_bits_requested", **metadata})
        state["bit_requests"] += 1
        state["bits_requested"] += count
        state["phase"] = "random_bits_source"
        value = random_bits(count)
        state["bit_values_obtained"] += 1
        if type(value) is not int or not 0 <= value < (1 << count):
            raise ValueError("random bit source returned an invalid integer")
        state["phase"] = "random_bits_journal"
        record({"kind": "random_bits_returned", **metadata, "value_decimal": str(value)})
        state["bit_values_journaled"] += 1
        state["phase"] = "sampling"
        return value

    try:
        if top_k > pipeline.model_vocabulary_size:
            raise ValueError("top-k exceeds vocabulary")
        started = {"kind": "response_started", "prompt_length": len(prompt), "prompt_sha256": digest(prompt),
                   "condition": condition, "purpose": purpose, "max_tokens": max_tokens,
                   "temperature": temperature, "top_k": top_k, "prefill_step_size": prefill_step_size,
                   "max_random_draws": max_random_draws, "max_random_bits": max_random_bits,
                   "max_model_calls": max_model_calls, "capture_public_fixture": capture_public_fixture, "shared_filter_settings": settings,
                   "runtime_profile_sha256": runtime, "score_namespace_sha256": namespace,
                   "caller_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        if capture_public_fixture:
            started["public_fixture_prompt_ids"] = list(prompt)
        if constraint is not None:
            started["constraint"] = constraint.identity
        record(started)
        if max_tokens:
            state["phase"] = "cache_creation"
            reserve("cache_creation", {})
            if backend is None:
                import mlx.core as backend
            if cache_factory is None:
                from mlx_lm.models.cache import make_prompt_cache
                cache_factory = make_prompt_cache
            cache = cache_factory(model)
            ids = backend.array(prompt, dtype=backend.int32)
            offset = 0
            while len(ids) - offset > 1:
                count = min(prefill_step_size, len(ids) - offset - 1)
                forward(ids[offset:offset + count], prefill=True)
                backend.eval([entry.state for entry in cache])
                offset += count
            next_input = ids[offset:]
            for index in range(max_tokens):
                output = forward(next_input, prefill=False)
                state["phase"] = "raw_model_head"
                head = output[:, -1, :].astype(backend.float32)
                backend.eval(head)
                filtered = np.array(head)
                state["phase"] = "sample_reservation"
                reserve("sample", {"index": index})
                finite = np.isfinite(filtered[0])
                prepared = {"kind": "prepared_step", "index": index,
                            "raw_finite_count": int(np.count_nonzero(finite)), "input_stage": "raw_model_head_before_shared_filter", "raw_logits_sha256": hashlib.sha256(filtered.tobytes()).hexdigest()}
                if capture_public_fixture:
                    support = np.flatnonzero(finite)
                    prepared.update({"public_fixture_raw_support": support.tolist(),
                                     "public_fixture_raw_logits": filtered[0, support].tolist()})
                record(prepared)
                if constraint is not None:
                    state["phase"] = "grammar_mask"
                    if np.isnan(filtered).any() or np.isposinf(filtered).any():
                        raise ValueError("model returned invalid logits")
                    allowed = constraint.allowed()
                    mask = np.zeros(pipeline.model_vocabulary_size, dtype=bool)
                    mask[:constraint.size] = allowed
                    filtered[0, ~mask] = -np.inf
                    record({"kind": "grammar_mask", "index": index,
                            "allowed_sha256": hashlib.sha256(allowed.tobytes()).hexdigest(),
                            "allowed_count": int(allowed.sum()),
                            "masked_logits_sha256": hashlib.sha256(filtered.tobytes()).hexdigest()})
                    reserve("after_grammar_mask", {"index": index})
                state["sample_attempts"] += 1
                state["phase"] = "sampling"
                step = pipeline.step(filtered, bits)
                state["sampled_tokens"] = len(pipeline.committed_token_ids)
                state["phase"] = "committed_step_journal"
                committed = {"kind": "committed_step", "index": index, "stopped": step.stopped is not None}
                if capture_public_fixture:
                    committed["public_fixture_token_id"] = step.token_id
                if constraint is not None:
                    committed["constraint_token_id"] = step.token_id
                record(committed)
                if constraint is not None:
                    state["phase"] = "grammar_commit"
                    constraint.commit(step.token_id)
                if step.stopped is not None:
                    response = asdict(step.stopped)
                    finish_constraint(response, "eos")
                    state["phase"] = "terminal_journal"
                    result = snapshot("eos")
                    record(result)
                    return {**result, "response": response}
                next_input = backend.array([step.token_id], dtype=backend.int32)
        state["phase"] = "finish"
        response = asdict(pipeline.finish_at_limit(max_tokens))
        finish_constraint(response, "length")
        state["phase"] = "terminal_journal"
        result = snapshot("length")
        record(result)
        return {**result, "response": response}
    except BaseException as exc:
        pipeline.close()
        failure = snapshot("error" if isinstance(exc, Exception) else "interrupted", type(exc).__name__)
        if not journal.broken:
            try:
                record(failure)
            except BaseException:
                failure["journal_broken"] = True
        raise ResponseFailure(failure) from exc
    finally:
        pipeline.close()
        cache = None
