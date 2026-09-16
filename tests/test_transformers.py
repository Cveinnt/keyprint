"""Runner contract tests use controlled models, not downloaded model behavior."""
import json
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from keyprint import Keyprint, KeyprintError
from keyprint.backends.bytelevel import ByteLevelBinding
from keyprint.backends.transformers import TransformersModel


class Tokenizer:
    def apply_chat_template(self, *args, **kwargs):
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
