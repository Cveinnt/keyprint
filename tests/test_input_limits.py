"""Input limits must be actionable without exposing arbitrary backend errors."""
import json
from types import SimpleNamespace

import pytest

from keyprint import InputLimitError
from keyprint.backends.mlx import MLXModel


@pytest.mark.parametrize("input_tokens", [128, 129])
def test_lowering_response_budget_cannot_fix_an_input_that_fills_context(input_tokens):
    error = InputLimitError(input_tokens=input_tokens, limit=128, max_tokens=32)
    assert str(error).endswith("Shorten the input.")
    assert "lower max_tokens" not in str(error)


def test_portable_context_budget_is_checked_before_inference_and_artifacts(tmp_path):
    from test_llama_cpp import candidate, Model
    model = Model()
    kp = candidate(model)
    with pytest.raises(InputLimitError) as caught:
        kp.generate("hello", max_tokens=128, output=tmp_path / "rejected")
    assert (caught.value.input_tokens, caught.value.max_tokens, caught.value.limit) == (1, 128, 128)
    assert model.calls == model.resets == 0
    assert not (tmp_path / "rejected").exists()
    # Exactly filling the declared context is legal; no smaller hidden limit.
    result = kp.generate("hello", max_tokens=127, output=tmp_path / "accepted")
    assert result.text == "A" and model.calls == 2


@pytest.mark.parametrize("ids", [[], [True], [4]])
def test_invalid_binding_is_not_misreported_as_input_overflow(ids, tmp_path):
    from test_llama_cpp import candidate
    kp = candidate()
    kp._backend.encode_prompt = lambda _: ids
    with pytest.raises(ValueError) as caught:
        kp.generate("hello", max_tokens=128, output=tmp_path / "rejected")
    assert not isinstance(caught.value, InputLimitError)
    assert not (tmp_path / "rejected").exists()


def test_mlx_input_only_limit_preserves_boundary_and_binding_validation():
    ids = [1] * 8192
    model = MLXModel(None, SimpleNamespace(apply_chat_template=lambda *a, **kw: ids))
    assert model.encode_prompt("hello") == ids
    ids.append(1)
    with pytest.raises(InputLimitError) as caught:
        model.encode_prompt("hello")
    assert (caught.value.input_tokens, caught.value.limit, caught.value.max_tokens) == (8193, 8192, None)
    ids[0] = True
    with pytest.raises(ValueError) as invalid:
        model.encode_prompt("hello")
    assert not isinstance(invalid.value, InputLimitError)


@pytest.mark.parametrize("params", [
    {"input_tokens": "private", "limit": 10},
    {"input_tokens": True, "limit": 10},
    {"input_tokens": 11, "limit": 0},
    {"input_tokens": 11, "limit": 10, "max_tokens": False},
    {"input_tokens": 1, "limit": 10, "max_tokens": 9},
])
def test_safe_error_rejects_invalid_counts(params):
    with pytest.raises(ValueError) as caught:
        InputLimitError(**params)
    assert not isinstance(caught.value, InputLimitError)
    assert "private" not in str(caught.value)


@pytest.mark.parametrize("protocol", ["openai", "anthropic"])
@pytest.mark.parametrize("typed", [True, False])
def test_api_error_replay_and_recovery_preserve_redaction(tmp_path, protocol, typed):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from keyprint.server import create_app
    from test_server import Model, TOKEN, HEADERS, BODY

    class Limited(Model):
        attempts = 0
        def generate(self, prompt, **kwargs):
            self.attempts += 1
            if prompt == "too long":
                if typed:
                    raise InputLimitError(input_tokens=120, max_tokens=32, limit=128)
                raise ValueError("private path and secret key")
            return super().generate(prompt, **kwargs)

    model = Limited()
    route = "/v1/messages" if protocol == "anthropic" else "/v1/chat/completions"
    headers = {**HEADERS, "Idempotency-Key": "overflow", "anthropic-version": "2023-06-01"}
    body = {**BODY, "max_tokens": 32, "messages": [{"role": "user", "content": "too long"}]}
    with TestClient(create_app(lambda: model, api_key=TOKEN, output=tmp_path / "runs")) as client:
        first = client.post(route, headers=headers, json=body)
        assert first.status_code == 400
        message = first.json()["error"]["message"]
        assert "private" not in message and "secret" not in message
        assert ("120" in message and "128" in message and "lower max_tokens" in message) is typed
        assert client.post(route, headers=headers, json=body).json() == first.json()
        assert model.attempts == 1 and model.calls == []
        assert not list((tmp_path / "runs").rglob("journal.jsonl"))
        good = client.post(route, headers={**headers, "Idempotency-Key": "recovery"},
                           json={**BODY, "max_tokens": 32})
        assert good.status_code == 200 and model.attempts == 2


@pytest.mark.parametrize("action", ["generate"])
def test_playground_limit_preserves_result_replays_and_recovers(tmp_path, action):
    pytest.importorskip("fastapi")
    from test_playground import Model, client_for, BODY, HEADERS

    class Limited(Model):
        attempts = 0
        def generate(self, prompt, **kwargs):
            self.attempts += 1
            if "oversized" in prompt:
                raise InputLimitError(input_tokens=120, max_tokens=32, limit=128)
            return super().generate(prompt, **kwargs)

    instances = []
    def factory():
        model = Limited()
        instances.append(model)
        return model

    with client_for(tmp_path, factory) as client:
        first = client.post("/api/experiment", headers=HEADERS, json=BODY)
        assert first.status_code == 200
        headers = {**HEADERS, "Idempotency-Key": "overflow"}
        body = {"action": action, "text": "An oversized source.", "max_tokens": 32}
        rejected = client.post("/api/experiment", headers=headers, json=body)
        assert rejected.status_code == 400
        assert "lower max_tokens" in rejected.json()["error"]["message"]
        attempts = instances[0].attempts
        assert client.post("/api/experiment", headers=headers, json=body).json() == rejected.json()
        assert instances[0].attempts == attempts
        assert client.get("/api/session", headers=HEADERS).json()["latest"] == first.json()
        assert client.get("/api/progress", headers=HEADERS).json() == {"active": False}
        assert client.post("/api/experiment", headers={**HEADERS, "Idempotency-Key": "recovery"},
                           json=BODY).status_code == 200
    statuses = [json.loads(p.read_text())["http_status"] for p in (tmp_path / "runs").rglob("status.json")]
    assert sorted(statuses) == [200, 200, 400]
