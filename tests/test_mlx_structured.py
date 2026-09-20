"""Exercise the pinned caller and real grammar with controlled model heads."""
import json
from threading import Event

import numpy as np
import pytest

pytest.importorskip("llguidance")
pytest.importorskip("jsonschema")

from keyprint import Keyprint, KeyprintError, KeyprintCancelled
from keyprint.backends.mlx import MLXModel
from keyprint.structured import JsonConstraint
from keyprint._engine.legacy._impl.research.grouped_tokenizer_binding import Binding
from test_fast_caller import Backend


SCHEMA = {"type": "object", "properties": {"name": {"type": "string"}},
          "required": ["name"], "additionalProperties": False}


class Tokenizer:
    def __init__(self):
        self.raw = Binding()._tokenizer

    def encode(self, text, **kwargs):
        return self.raw.encode(text, **kwargs).ids

    def apply_chat_template(self, *args, **kwargs):
        return [32]


def candidate(monkeypatch, text='{"name":"Maya"}', execution="reference"):
    tokenizer = Tokenizer()
    ids = tokenizer.encode(text, add_special_tokens=False) + [151645]
    calls = []

    def model(inputs, **kwargs):
        token = ids[len(calls)]
        calls.append(token)
        raw = np.full((1, inputs.shape[1], 151936), -np.inf, dtype=np.float32)
        # Padding and channel controls must never win. Premature EOS must be
        # rejected, even though all three outrank the requested JSON token.
        raw[:, :, [151900, 151667]] = 1e6
        raw[:, :, 151645] = 1e4
        raw[:, :, token] = 0
        return raw

    monkeypatch.setattr(MLXModel, "load", lambda _: MLXModel(model, tokenizer))
    kp = Keyprint.from_mlx("unused", key=bytes(range(32)), execution=execution)
    run = kp._run
    kp._run = lambda *a, **kw: run(*a, **kw, backend=Backend, cache_factory=lambda _: [])
    return kp, ids, calls


@pytest.mark.parametrize("execution", ["reference", "experimental-fast"])
@pytest.mark.parametrize("condition", ["ordinary", "marked"])
def test_both_mlx_callers_constrain_before_filter_and_replay_every_commit(monkeypatch, tmp_path, execution, condition):
    text = '{"name":"Maya é中🙂"}'
    kp, ids, calls = candidate(monkeypatch, text, execution)
    result = kp.generate("JSON", json_schema=SCHEMA, max_tokens=64,
                         condition=condition, output=tmp_path / "run")
    assert result.text == text and calls == ids
    assert result.report["structured_output"]["schema_validated"]
    retained = json.loads((result.artifacts / "report.json").read_text())
    assert retained["reservations"]["sample"] == len(ids)
    assert retained["reservations"]["after_grammar_mask"] == len(ids)
    events = [json.loads(line)["event"] for line in (result.artifacts / "journal.jsonl").read_text().splitlines()]
    constraint = JsonConstraint.for_mlx(SCHEMA, kp._backend.tokenizer)
    assert events[0]["constraint"] == constraint.identity
    masks = 0
    for event in events:
        if event["kind"] == "grammar_mask":
            masks += 1
            allowed = constraint.allowed()
        elif event["kind"] == "committed_step":
            token = event["constraint_token_id"]
            assert allowed[token]
            constraint.commit(token)
    assert masks == len(ids) == result.report["usage"]["completion_tokens"]
    assert constraint.finish(text, "eos")["schema_validated"]


def test_length_is_explicitly_incomplete_without_extra_forward(monkeypatch, tmp_path):
    kp, _, calls = candidate(monkeypatch)
    result = kp.generate("JSON", json_schema=SCHEMA, max_tokens=1, output=tmp_path / "run")
    assert len(calls) == 1 and result.report["payload"]["completion"] == "length"
    assert result.report["structured_output"]["status"] == "incomplete"
    assert not result.report["structured_output"]["schema_validated"]


def test_schema_rejected_before_forward_or_journal(monkeypatch, tmp_path):
    kp, _, calls = candidate(monkeypatch)
    with pytest.raises(ValueError, match="unsupported"):
        kp.generate("JSON", json_schema={"type": "object", "$ref": "https://example.com/schema"}, output=tmp_path / "run")
    assert not calls and not (tmp_path / "run").exists()


def test_mask_failure_retains_uncommitted_attempt(monkeypatch, tmp_path):
    kp, _, calls = candidate(monkeypatch)
    def fail(self):
        raise ValueError("injected mask failure")
    monkeypatch.setattr(JsonConstraint, "allowed", fail)
    with pytest.raises(KeyprintError) as caught:
        kp.generate("JSON", json_schema=SCHEMA, output=tmp_path / "run")
    assert len(calls) == 1
    assert caught.value.report["payload"]["phase"] == "grammar_mask"
    assert caught.value.report["usage"]["completion_tokens"] == 0
    assert not caught.value.report["structured_output"]["schema_validated"]


def test_validation_failure_is_not_a_completed_response(monkeypatch, tmp_path):
    kp, _, _ = candidate(monkeypatch)
    def fail(*args):
        raise ValueError("injected independent validation failure")
    monkeypatch.setattr(JsonConstraint, "finish", fail)
    with pytest.raises(KeyprintError) as caught:
        kp.generate("JSON", json_schema=SCHEMA, output=tmp_path / "run")
    assert caught.value.report["payload"]["phase"] == "structured_validation"
    assert caught.value.report["usage"]["completion_tokens"] > 0
    assert not caught.value.report["structured_output"]["schema_validated"]


def test_cancel_after_mask_before_sample_retains_zero_draws(monkeypatch, tmp_path):
    kp, _, calls = candidate(monkeypatch)
    stop = Event()
    original = JsonConstraint.allowed
    def stop_after_mask(self):
        value = original(self)
        stop.set()
        return value
    monkeypatch.setattr(JsonConstraint, "allowed", stop_after_mask)
    with pytest.raises(KeyprintCancelled) as caught:
        kp.generate("JSON", json_schema=SCHEMA, cancel_event=stop, output=tmp_path / "run")
    assert len(calls) == 1
    assert caught.value.report["payload"]["bit_requests"] == 0
    assert caught.value.report["usage"]["completion_tokens"] == 0
