"""Runner contract tests use controlled models, not downloaded model behavior."""
import json
from types import SimpleNamespace
from threading import Event

import pytest

torch = pytest.importorskip("torch")

from keyprint import Keyprint, KeyprintError, KeyprintCancelled
from keyprint.backends.bytelevel import ByteLevelBinding
from keyprint.backends.transformers import TransformersModel


class Tokenizer:
    def apply_chat_template(self, *args, **kwargs):
        self.template_kwargs = kwargs
        return [1]

    def decode(self, ids, **kwargs):
        return "".join("A" if i == 1 else "" for i in ids)

    def encode(self, text, **kwargs):
        return [1] * len(text)


class Model:
    config = SimpleNamespace(max_position_embeddings=100)
    def __init__(self, *, fail=False):
        self.calls = 0
        self.fail = fail
        self.inputs = []

    def __call__(self, *, input_ids, past_key_values, use_cache):
        self.inputs.append((input_ids.tolist(), past_key_values))
        self.calls += 1
        if self.fail:
            raise RuntimeError("fixture failure")
        raw = torch.full((1, input_ids.shape[1], 4), -torch.inf, dtype=torch.float32)
        raw[:, :, 1 if self.calls == 1 else 0] = 0
        raw[:, :, 2:] = 99  # forbidden control and padding must be excluded
        return SimpleNamespace(logits=raw, past_key_values="cache")


def candidate(model):
    serialized = json.dumps({"model": {"type": "BPE", "vocab": {"<eos>": 0, "A": 1, "<control>": 2}},
        "decoder": {"type": "ByteLevel"}, "normalizer": None,
        "added_tokens": [{"id": 0, "content": "<eos>", "special": True},
                         {"id": 2, "content": "<control>", "special": True}]})
    binding = ByteLevelBinding.create(serialized, vocabulary_size=4, special_ids=[0, 2], eos_ids=[0])
    kp = Keyprint(key=bytes(range(32)))
    kp._backend = TransformersModel(model, Tokenizer(), binding, {}, temperature=.7, top_k=4)
    return kp


@pytest.mark.parametrize("condition", ["ordinary", "marked"])
def test_cache_eos_controls_and_receipts(condition, tmp_path):
    model = Model()
    result = candidate(model).generate("hello", condition=condition, max_tokens=4, output=tmp_path / "run")
    assert result.text == "A"
    assert result.report["committed_token_ids"] == [1, 0]
    assert result.report["completion"] == "eos"
    assert model.calls == 2
    assert model.inputs == [([[1]], None), ([[1]], "cache")]
    records = [json.loads(line)["event"] for line in (result.artifacts / "journal.jsonl").read_text().splitlines()]
    phases = [record["phase"] for record in records]
    assert phases[-1] == "complete"
    assert phases.count("committed") == 2
    assert phases.index("commit_requested") < phases.index("committed")
    assert result.report["verdict"] is None
    assert result.report["identity"]["empirical_acceptance_transfers"] is False


def test_visible_text_requests_non_thinking_template_and_records_policy(tmp_path):
    kp = candidate(Model())
    result = kp.generate("hello", max_tokens=4, output=tmp_path / "run")
    assert kp._backend.tokenizer.template_kwargs["enable_thinking"] is False
    assert result.report["identity"]["chat_template_kwargs"] == {"enable_thinking": False}
    events = [json.loads(line)["event"] for line in (result.artifacts / "journal.jsonl").read_text().splitlines()]
    assert events[0]["identity"]["chat_template_kwargs"] == {"enable_thinking": False}


def test_failure_does_not_retry_and_retains_attempt(tmp_path):
    model = Model(fail=True)
    with pytest.raises(KeyprintError) as caught:
        candidate(model).generate("hello", output=tmp_path / "failed")
    assert model.calls == 1
    report = caught.value.report
    assert report["failed_phase"] == "model_forward"
    assert report["model_calls"] == 1
    assert report["committed_token_ids"] == []
    assert json.loads((caught.value.artifacts / "report.json").read_text()) == report


def test_wrong_binding_cannot_use_reference_pipeline():
    with pytest.raises(ValueError, match="reference binding"):
        candidate(Model()).pipeline()


def test_literal_is_uncalibrated_and_rejects_render_change():
    kp = candidate(Model())
    assert kp.score("AAA")["verdict"] is None
    with pytest.raises(ValueError, match="exactly replayable"):
        kp.score("BBB")


@pytest.mark.parametrize("cancel_at", [0, 1, 2])
def test_cooperative_cancel_retains_partial_usage_and_allows_new_attempt(tmp_path, cancel_at):
    stop = Event()
    class CancellingModel(Model):
        def __call__(self, **kwargs):
            result = super().__call__(**kwargs)
            if self.calls == cancel_at:
                stop.set()
            return result
    model = CancellingModel()
    kp = candidate(model)
    if cancel_at == 0:
        stop.set()
    with pytest.raises(KeyprintCancelled) as caught:
        kp.generate("hello", max_tokens=4, cancel_event=stop, output=tmp_path / "cancelled")
    report = caught.value.report
    assert model.calls == cancel_at
    assert report["model_calls"] == cancel_at
    assert report["committed_token_ids"] == ([1] if cancel_at == 2 else [])
    assert report["usage"]["completion_tokens"] == (1 if cancel_at == 2 else 0)
    assert report["cancellation_requested"] is True
    assert json.loads((caught.value.artifacts / "report.json").read_text()) == report
    events = [json.loads(line)["event"] for line in (caught.value.artifacts / "journal.jsonl").read_text().splitlines()]
    assert [event["token_id"] for event in events if event["phase"] == "committed"] == report["committed_token_ids"]
    result = kp.generate("hello", max_tokens=4, output=tmp_path / "next")
    assert result.report["completion"] == "eos"


def test_invalid_cancel_event_rejected_before_model(tmp_path):
    model = Model()
    with pytest.raises(TypeError, match="threading.Event"):
        candidate(model).generate("hello", cancel_event=True, output=tmp_path / "bad")
    assert model.calls == 0 and not (tmp_path / "bad").exists()


def test_cancellation_does_not_mask_a_model_failure(tmp_path):
    stop = Event()
    class FailingModel(Model):
        def __call__(self, **kwargs):
            stop.set()
            return super().__call__(**kwargs)
    with pytest.raises(KeyprintError) as caught:
        candidate(FailingModel(fail=True)).generate("hello", cancel_event=stop, output=tmp_path / "failed")
    assert not isinstance(caught.value, KeyprintCancelled)
    assert caught.value.report["error_type"] == "RuntimeError"


def test_cancellation_during_random_draw_finishes_that_token_commit(tmp_path, monkeypatch):
    stop = Event()
    class TwoTokens(Model):
        def __call__(self, **kwargs):
            result = super().__call__(**kwargs)
            result.logits[:, :, 0:2] = 0
            return result
    def draw(bits):
        stop.set()
        return (1 << bits) - 1
    monkeypatch.setattr("keyprint.backends.transformers.secrets.randbits", draw)
    model = TwoTokens()
    with pytest.raises(KeyprintCancelled) as caught:
        candidate(model).generate("hello", condition="ordinary", cancel_event=stop, output=tmp_path / "stopped")
    assert model.calls == 1
    assert caught.value.report["committed_token_ids"] == [1]
    events = [json.loads(line)["event"] for line in (caught.value.artifacts / "journal.jsonl").read_text().splitlines()]
    phases = [event["phase"] for event in events]
    assert phases.index("random_returned") < phases.index("commit_requested") < phases.index("committed")
