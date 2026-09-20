import json
import hashlib
import math
import random
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest

from keyprint import Keyprint, KeyprintCancelled, KeyprintError
from keyprint.experimental.fast_public import FastPublicCandidate
from keyprint._engine.research import keyprint_candidate_v3_caller as journal_module


class Backend:
    array = staticmethod(np.array)
    int32, float32 = np.int32, np.float32
    eval = staticmethod(lambda _: None)


def candidate():
    result = Keyprint(key=bytes(range(32)))
    result._candidate = FastPublicCandidate(result._candidate)
    return result


def head(ids, tokens=(32, 33, 34)):
    result = np.full((1, ids.shape[1], 151936), -np.inf, dtype=np.float32)
    result[:, :, list(tokens)] = 0
    return result


@pytest.mark.parametrize('capture', [False, True])
@pytest.mark.parametrize('pattern', ['sparse', 'dense', 'padding'])
def test_raw_head_count_and_fixture_indices_preserve_journal_contract(tmp_path, capture, pattern):
    raw = head(np.array([[32]]))
    if pattern == 'dense': raw[:] = 0.
    elif pattern == 'padding': raw[0, 0, -1] = 1.
    raw[0, 0, 32] = 100.
    path = tmp_path / 'journal.jsonl'
    with journal_module.DurableJournal(path) as journal:
        candidate()._candidate.run_response(
            lambda *_args, **_kwargs: raw.copy(), [32], key=bytes(range(32)),
            condition='ordinary', random_bits=lambda _count: 0, journal=journal,
            reserve=lambda *_args: None, max_tokens=1, allow_thinking=False,
            capture_public_fixture=capture, backend=Backend, cache_factory=lambda _: [])
    events = [json.loads(line)['event'] for line in path.read_text().splitlines()]
    prepared, = [e for e in events if e['kind'] == 'prepared_step']
    expected_ids = [i for i, value in enumerate(raw[0, 0]) if math.isfinite(float(value))]
    assert prepared['raw_finite_count'] == len(expected_ids)
    assert type(prepared['raw_finite_count']) is int
    assert prepared['raw_logits_sha256'] == hashlib.sha256(raw.tobytes()).hexdigest()
    if capture:
        assert prepared['public_fixture_raw_support'] == expected_ids
        assert prepared['public_fixture_raw_logits'] == [float(raw[0, 0, i]) for i in expected_ids]
    else:
        assert not any(name.startswith('public_fixture_') for name in prepared)


@pytest.mark.parametrize("condition", ["ordinary", "marked"])
def test_complete_caller_probabilities_draws_text_and_inspection_match(tmp_path, monkeypatch, condition):
    original, fast = Keyprint(key=bytes(range(32))), candidate()
    results = []
    for name, kp in (("reference", original), ("fast", fast)):
        monkeypatch.setattr("keyprint.api.secrets.randbits", random.Random(17).getrandbits)
        results.append(kp._run(lambda ids, **_: head(ids), [32], max_tokens=5, condition=condition,
                               output=tmp_path / name, backend=Backend, cache_factory=lambda _: []))
    a, b = results
    assert a.text == b.text
    assert a.report["payload"]["sampling_records"] == b.report["payload"]["sampling_records"]
    assert a.report["usage"] == b.report["usage"]
    assert a.report["target_identity"] != b.report["target_identity"]
    assert "Experimental" in b.report["package_scope"]
    assert original.inspect(a.text).fraction == fast.inspect(b.text).fraction
    assert b.report["schema"] == "keyprint.experimental-fast-report.v1"


@pytest.mark.parametrize("cancel_at", [0, 1, 2])
def test_cancel_retains_work_and_reuses_same_bound_candidate(tmp_path, cancel_at):
    stop, calls = Event(), []
    def model(ids, **_):
        calls.append(1)
        if len(calls) == cancel_at: stop.set()
        return head(ids, (32,))
    kp = candidate()
    bound = kp._candidate._core.reference
    if cancel_at == 0: stop.set()
    with pytest.raises(KeyprintCancelled) as caught:
        kp._run(model, [32], max_tokens=4, condition="marked", output=tmp_path / "cancelled",
                backend=Backend, cache_factory=lambda _: [], cancel_event=stop)
    report = caught.value.report
    assert len(calls) == cancel_at
    assert report["usage"]["completion_tokens"] == (1 if cancel_at == 2 else 0)
    assert report["cancellation_requested"] is True
    assert json.loads((caught.value.artifacts / "report.json").read_text())["report"] == report
    after = kp._run(model, [32], max_tokens=2, condition="marked", output=tmp_path / "next",
                    backend=Backend, cache_factory=lambda _: [])
    assert after.text == "AA"
    assert kp._candidate._core.reference is bound


def test_draw_journal_failure_does_not_commit_or_become_cancellation(tmp_path, monkeypatch):
    class BrokenJournal(journal_module.DurableJournal):
        def append(self, event):
            if event["kind"] == "random_bits_returned":
                self.broken = True
                raise OSError("injected disk failure")
            return super().append(event)
    monkeypatch.setattr(journal_module, "DurableJournal", BrokenJournal)
    with pytest.raises(KeyprintError) as caught:
        candidate()._run(lambda ids, **_: head(ids), [32], max_tokens=2, condition="ordinary",
                         output=tmp_path / "failed", backend=Backend, cache_factory=lambda _: [])
    assert type(caught.value) is KeyprintError
    payload = caught.value.report["payload"]
    assert payload["code"] == "journal_failure"
    assert payload["committed_tokens"] == 0 and payload["bit_values_obtained"] == 1


def test_execution_opt_in_is_validated_before_model_loading(monkeypatch):
    calls = []
    monkeypatch.setattr("keyprint.backends.mlx.MLXModel.load", lambda p: calls.append(p) or SimpleNamespace())
    with pytest.raises(ValueError, match="execution"):
        Keyprint.from_mlx("unused", key=bytes(range(32)), execution="unknown")
    assert not calls
    fast = Keyprint.from_mlx("unused", key=bytes(range(32)), execution="experimental-fast")
    assert len(calls) == 1 and fast.identity["version"] == "keyprint-mlx-sparse-experimental-v1"


@pytest.mark.parametrize("mutation", ["version", "digest", "reference", "source", "namespace"])
def test_reporting_rejects_wrong_execution_identity(mutation):
    from keyprint.experimental.fast_reporting import _experimental_target, _digest
    identity = candidate().identity
    if mutation == "version":
        identity["version"] = "keyprint-candidate-v3-shared-support-2026-09-09"
    elif mutation == "digest":
        identity["runtime_profile_sha256"] = "0" * 64
    else:
        if mutation == "reference":
            identity["specification"]["reference_runtime_sha256"] = "0" * 64
        elif mutation == "source":
            identity["specification"]["execution"]["sources"]["fast_caller.py"] = "0" * 64
        else:
            identity["score_namespace_sha256"] = "0" * 64
        identity["runtime_profile_sha256"] = _digest(identity["specification"])
    with pytest.raises(ValueError):
        _experimental_target(identity)


def test_experimental_reporting_does_not_weaken_frozen_contract():
    from keyprint.experimental.fast_reporting import _experimental_target
    from keyprint._engine.research.keyprint_v3_reporting_contract import _target
    reference = Keyprint(key=bytes(range(32))).identity
    fast = candidate().identity
    _target(reference)
    _experimental_target(fast)
    with pytest.raises(ValueError):
        _target(fast)
    with pytest.raises(ValueError):
        _experimental_target(reference)



def test_fast_supplied_head_pipeline_and_verification_use_experimental_reports():
    kp = candidate()
    with kp.pipeline() as pipeline:
        step = pipeline.step(head(np.array([[32]]), (32,))[0, -1:, :], lambda _: 0)
        assert step["kind"] == "generation_trace"
        assert step["latest_emitted_text"] == "A"
        final = pipeline.finish()
        assert final["rendered_carriers"]["visible_text"] == "A"
        assert pipeline.receipt() == final
    payload = {"status": "pass", "artifact_sha256": "a" * 64,
               "evaluated_runtime_profile_sha256": kp.identity["runtime_profile_sha256"],
               "checks": [{"name": "fixture", "status": "pass"}]}
    report = kp._candidate.project_verification(payload)
    assert report["kind"] == "verification"
    assert report["schema"] == "keyprint.experimental-fast-report.v1"
    assert report["verdict"] is None
    payload["evaluated_runtime_profile_sha256"] = "0" * 64
    assert kp._candidate.project_verification(payload)["kind"] == "error"
