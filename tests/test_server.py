import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient
from keyprint import Generation
from keyprint.server import create_app

TOKEN = "local-test-token-" * 3
HEADERS = {"Authorization": "Bearer " + TOKEN}
BODY = {"model": "keyprint", "messages": [{"role": "user", "content": "hello"}]}


class Model:
    def __init__(self, *, block=False, fail=False):
        self.calls = []
        self.started, self.release = threading.Event(), threading.Event()
        self.block, self.fail = block, fail

    def generate(self, prompt, *, max_tokens, output):
        self.calls.append(prompt)
        self.started.set()
        if self.block:
            assert self.release.wait(10)
        if self.fail:
            raise RuntimeError("private failure that must not reach the client")
        Path(output).mkdir(mode=0o700)
        return Generation(prompt, {"completion": "eos", "usage": {
            "prompt_tokens": 4, "completion_tokens": 1, "total_tokens": 5}}, output)


def test_authentication_and_exact_usage(tmp_path):
    model = Model()
    with TestClient(create_app(lambda: model, api_key=TOKEN, output=tmp_path / "runs")) as client:
        assert client.post("/v1/chat/completions", json=BODY).status_code == 401
        assert model.calls == []
        response = client.post("/v1/chat/completions", json=BODY, headers=HEADERS)
        assert response.status_code == 200
        data = response.json()
        assert data["choices"][0]["message"]["content"] == "hello"
        assert data["usage"]["total_tokens"] == 5
        assert data["keyprint"]["hosted_provider"] is False
        assert "private" not in response.text and TOKEN not in response.text


@pytest.mark.parametrize("extra", [{"stream": True}, {"tools": []}, {"logprobs": True},
    {"temperature": .5}, {"n": 2}, {"n": True}, {"stream": 0}, {"max_tokens": True}, {"max_tokens": 1, "max_completion_tokens": 1},
    {"messages": [{"role": "system", "content": "x"}]}, {"model": "gpt-anything"}])
def test_unsupported_modes_fail_before_model(extra, tmp_path):
    model = Model()
    with TestClient(create_app(lambda: model, api_key=TOKEN, output=tmp_path / "runs")) as client:
        assert client.post("/v1/chat/completions", json={**BODY, **extra}, headers=HEADERS).status_code == 400
        assert model.calls == []


def test_idempotency_and_budget(tmp_path):
    model = Model()
    with TestClient(create_app(lambda: model, api_key=TOKEN, output=tmp_path / "runs", max_requests=1)) as client:
        headers = {**HEADERS, "Idempotency-Key": "one"}
        first = client.post("/v1/chat/completions", json=BODY, headers=headers)
        second = client.post("/v1/chat/completions", json=BODY, headers=headers)
        assert first.json() == second.json()
        changed = {**BODY, "max_tokens": 2}
        assert client.post("/v1/chat/completions", json=changed, headers=headers).status_code == 409
        assert client.post("/v1/chat/completions", json=BODY, headers=HEADERS).status_code == 429
        assert model.calls == ["hello"]


def test_single_flight_prevents_request_interleaving(tmp_path):
    model = Model(block=True)
    with TestClient(create_app(lambda: model, api_key=TOKEN, output=tmp_path / "runs")) as client:
        with ThreadPoolExecutor(max_workers=1) as threads:
            first = threads.submit(client.post, "/v1/chat/completions", json=BODY, headers=HEADERS)
            try:
                assert model.started.wait(5)
                second = client.post("/v1/chat/completions", json=BODY, headers=HEADERS)
                assert second.status_code == 503
            finally:
                model.release.set()
            assert first.result().status_code == 200
        assert model.calls == ["hello"]


def test_failed_attempt_not_retried_and_no_private_error_leak(tmp_path):
    model = Model(fail=True)
    with TestClient(create_app(lambda: model, api_key=TOKEN, output=tmp_path / "runs")) as client:
        headers = {**HEADERS, "Idempotency-Key": "failed"}
        for _ in range(2):
            response = client.post("/v1/chat/completions", json=BODY, headers=headers)
            assert response.status_code == 500
            assert response.headers["x-should-retry"] == "false"
            assert "private failure" not in response.text
        assert model.calls == ["hello"]


def test_size_limit(tmp_path):
    model = Model()
    with TestClient(create_app(lambda: model, api_key=TOKEN, output=tmp_path / "runs")) as client:
        response = client.post("/v1/chat/completions", content=b" " * 131073,
                               headers={**HEADERS, "Content-Type": "application/json"})
        assert response.status_code == 413 and model.calls == []


def test_model_loaded_and_used_on_same_worker(tmp_path):
    class OwnedModel(Model):
        def __init__(self):
            super().__init__()
            self.owner = threading.get_ident()

        def generate(self, *args, **kwargs):
            assert threading.get_ident() == self.owner
            return super().generate(*args, **kwargs)

    with TestClient(create_app(OwnedModel, api_key=TOKEN, output=tmp_path / "runs")) as client:
        assert client.post("/v1/chat/completions", json=BODY, headers=HEADERS).status_code == 200
