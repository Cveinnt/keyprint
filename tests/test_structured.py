import hashlib
import json
from threading import Event
from types import SimpleNamespace

import pytest

pytest.importorskip("llguidance")
pytest.importorskip("jsonschema")
torch = pytest.importorskip("torch")

from keyprint import Keyprint, KeyprintError, KeyprintCancelled
from keyprint.backends.bytelevel import ByteLevelBinding
from keyprint.backends.transformers import TransformersModel
from keyprint.structured import JsonConstraint

SCHEMA = {"type": "object", "properties": {"name": {"type": "string"},
          "count": {"type": "integer"}, "enabled": {"type": "boolean"}},
          "required": ["name", "count", "enabled"], "additionalProperties": False}
TEXT = '{"name":"Maya","count":3,"enabled":false}'


class Tokenizer:
    def encode(self, text, **kwargs):
        return [byte + 1 for byte in text.encode()]

    def decode(self, ids, **kwargs):
        return bytes(i - 1 for i in ids if i != 0).decode()

    def apply_chat_template(self, *args, **kwargs):
        return [66]


class Model:
    config = SimpleNamespace(max_position_embeddings=1024)

    def __init__(self, text=TEXT, cancel=None):
        self.tokens = Tokenizer().encode(text) + [0]
        self.calls = 0
        self.cancel = cancel

    def __call__(self, **kwargs):
        logits = torch.full((1, 1, 257), -20., dtype=torch.float32)
        logits[0, 0, self.tokens[min(self.calls, len(self.tokens) - 1)]] = 10
        logits[0, 0, 0] = 100  # EOS would win without a grammar mask before top-k.
        if self.calls == 0:
            logits[0, 0, ord('`') + 1] = 90  # Markdown fence would beat '{' too.
        self.calls += 1
        if self.cancel is not None:
            self.cancel.set()
        return SimpleNamespace(logits=logits, past_key_values=None)


def candidate(model=None):
    binding = ByteLevelBinding((None, *(bytes([b]) for b in range(256))),
                              frozenset([0]), hashlib.sha256(b"byte-fixture").hexdigest())
    kp = Keyprint(key=bytes(range(32)))
    kp._backend = TransformersModel(model or Model(), Tokenizer(), binding, {}, temperature=.7, top_k=1)
    return kp


@pytest.mark.parametrize("condition", ["ordinary", "marked"])
def test_grammar_precedes_top_k_and_watermark_with_exact_commits(condition, tmp_path):
    kp = candidate()
    result = kp.generate("Return JSON", json_schema=SCHEMA, condition=condition,
                         max_tokens=100, output=tmp_path / "run")
    assert result.text == TEXT
    assert result.report["structured_output"]["schema_validated"] is True
    events = [json.loads(line)["event"] for line in (result.artifacts / "journal.jsonl").read_text().splitlines()]
    assert events[0]["constraint"] == result.report["structured_output"]["identity"]
    tokens = [e["token_id"] for e in events if e["phase"] == "committed"]
    assert tokens == kp._backend.tokenizer.encode(TEXT) + [0]
    assert len([e for e in events if e["phase"] == "grammar_mask"]) == len(tokens)
    phases = [e["phase"] for e in events]
    assert phases.index("grammar_mask") < phases.index("prepared") < phases.index("commit_requested")


def test_unconstrained_path_keeps_model_distribution(tmp_path):
    result = candidate().generate("hello", max_tokens=100, output=tmp_path / "run")
    assert result.text == "" and result.report["committed_token_ids"] == [0]
    assert "structured_output" not in result.report


def test_capped_json_is_incomplete_not_repaired_or_retried(tmp_path):
    kp = candidate()
    result = kp.generate("JSON", json_schema=SCHEMA, max_tokens=5, output=tmp_path / "run")
    assert result.text == TEXT[:5] and kp._backend.model.calls == 5
    assert result.report["completion"] == "length"
    assert result.report["structured_output"]["schema_validated"] is False
    assert result.report["structured_output"]["status"] == "incomplete"


@pytest.mark.parametrize("schema", [[], {"type": "array"}, {"type": "object", "$ref": "https://example.com"},
    {"type": "object", "properties": {"x": {"type": "string", "pattern": "a+"}}},
    {"type": "object", "x-guidance": {"lenient": True}},
    {"type": "object", "properties": []}, {"type": "object", "required": "x"},
    {"type": "object", "description": "x" * 32768},
    {"type": "object", "minimum": float("nan")}])
def test_unsupported_schema_fails_before_inference(schema, tmp_path):
    kp = candidate()
    with pytest.raises(ValueError):
        kp.generate("JSON", json_schema=schema, output=tmp_path / "run")
    assert kp._backend.model.calls == 0 and not (tmp_path / "run").exists()


def test_non_transformers_backend_does_not_ignore_schema():
    kp = Keyprint(key=bytes(range(32)))
    kp._backend = object()
    with pytest.raises(ValueError, match="Transformers"):
        kp.generate("JSON", json_schema=SCHEMA)


@pytest.mark.parametrize("text", ['{"name":"é中🙂"}', '{"name":"quote: \\" slash: \\\\"}'])
def test_unicode_and_escaped_strings_are_not_postprocessed(text, tmp_path):
    schema = {"type": "object", "properties": {"name": {"type": "string"}},
              "required": ["name"], "additionalProperties": False}
    result = candidate(Model(text)).generate("JSON", json_schema=schema,
                    max_tokens=100, output=tmp_path / "run")
    assert result.text == text
    assert result.report["structured_output"]["schema_validated"]


def test_grammar_fault_retains_work_without_retry(tmp_path, monkeypatch):
    kp = candidate()
    def bad_mask(self):
        raise RuntimeError("fixture grammar failure")
    monkeypatch.setattr(JsonConstraint, "allowed", bad_mask)
    with pytest.raises(KeyprintError) as caught:
        kp.generate("JSON", json_schema=SCHEMA, output=tmp_path / "run")
    assert caught.value.report["failed_phase"] == "grammar_mask"
    assert caught.value.report["model_calls"] == 1
    assert caught.value.report["committed_token_ids"] == []
    assert kp._backend.model.calls == 1


def test_cancelled_structured_generation_reuses_worker_but_not_grammar_state(tmp_path):
    stop = Event()
    kp = candidate(Model(cancel=stop))
    with pytest.raises(KeyprintCancelled):
        kp.generate("JSON", json_schema=SCHEMA, output=tmp_path / "cancel", cancel_event=stop)
    kp._backend.model = Model()
    result = kp.generate("JSON", json_schema=SCHEMA, max_tokens=100, output=tmp_path / "next")
    assert result.text == TEXT and result.report["structured_output"]["schema_validated"]


def test_independent_validation_rejects_wrong_schema_even_if_matcher_accepts(tmp_path, monkeypatch):
    original = JsonConstraint.finish
    def check_other_schema(self, text, completion):
        from jsonschema import Draft202012Validator
        self.validator = Draft202012Validator({"const": {"wrong": True}})
        return original(self, text, completion)
    monkeypatch.setattr(JsonConstraint, "finish", check_other_schema)
    with pytest.raises(KeyprintError) as caught:
        candidate().generate("JSON", json_schema=SCHEMA, max_tokens=100, output=tmp_path / "run")
    assert caught.value.report["failed_phase"] == "structured_validation"
    assert caught.value.report["text"] == TEXT
    assert caught.value.report["structured_output"]["schema_validated"] is False
