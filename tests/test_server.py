import threading
import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient
from keyprint import Generation, KeyprintCancelled
from keyprint.server import create_app
from starlette.requests import Request

TOKEN = "local-test-token-" * 3
HEADERS = {"Authorization": "Bearer " + TOKEN}
BODY = {"model": "keyprint", "messages": [{"role": "user", "content": "hello"}]}


class Model:
    def __init__(self, *, block=False, fail=False):
        self.calls = []
        self.started, self.release = threading.Event(), threading.Event()
        self.block, self.fail = block, fail

    def generate(self, prompt, *, max_tokens, output, cancel_event=None):
        self.calls.append(prompt)
        self.started.set()
        if self.block:
            assert self.release.wait(10)
        if self.fail:
            raise RuntimeError("private failure that must not reach the client")
        if cancel_event is not None and cancel_event.is_set():
            raise KeyprintCancelled({"cancellation_requested": True}, Path(output))
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


@pytest.mark.parametrize("fail", [False, True])
def test_cancelled_http_handler_does_not_orphan_attempt_or_unlock_worker(tmp_path, fail):
    model = Model(block=True, fail=fail)
    app = create_app(lambda: model, api_key=TOKEN, output=tmp_path / "runs")
    endpoint = next(route.endpoint for route in app.routes if route.path == "/v1/chat/completions")

    def request(key, body=BODY):
        async def receive():
            return {"type": "http.request", "body": json.dumps(body).encode(), "more_body": False}
        return Request({"type": "http", "method": "POST", "path": "/v1/chat/completions",
                        "headers": [(b"content-type", b"application/json"), (b"idempotency-key", key.encode())]}, receive)

    async def exercise():
        async with app.router.lifespan_context(app):
            first = asyncio.create_task(endpoint(request("interrupted")))
            try:
                assert await asyncio.to_thread(model.started.wait, 5)
                first.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await first
                assert (await endpoint(request("interrupted"))).status_code == 409
                assert (await endpoint(request("other"))).status_code == 503
                model.release.set()
                async with asyncio.timeout(5):
                    while True:
                        recovered = await endpoint(request("interrupted"))
                        if recovered.status_code != 409:
                            break
                        await asyncio.sleep(.01)
                assert recovered.status_code == (500 if fail else 200)
                assert (await endpoint(request("interrupted"))).body == recovered.body
                assert (await endpoint(request("interrupted", {**BODY, "max_tokens": 3}))).status_code == 409
                assert model.calls == ["hello"]
                finished = list((tmp_path / "runs").glob("*-finished.json"))
                assert len(finished) == 1
                assert json.loads(finished[0].read_text())["http_status"] == recovered.status_code
            finally:
                model.release.set()
                if not first.done():
                    await first
    asyncio.run(exercise())


def test_record_failure_keeps_terminal_attempt_without_generation(tmp_path, monkeypatch):
    model = Model()
    original = Path.open
    def fail_start(path, *args, **kwargs):
        if path.name.endswith("-started.json"):
            raise OSError("private disk failure")
        return original(path, *args, **kwargs)
    with TestClient(create_app(lambda: model, api_key=TOKEN, output=tmp_path / "runs")) as client:
        monkeypatch.setattr(Path, "open", fail_start)
        headers = {**HEADERS, "Idempotency-Key": "disk-failed"}
        first = client.post("/v1/chat/completions", json=BODY, headers=headers)
        assert first.status_code == 500 and "private disk failure" not in first.text
        monkeypatch.setattr(Path, "open", original)
        assert client.post("/v1/chat/completions", json=BODY, headers=headers).json() == first.json()
        assert model.calls == []
        assert client.post("/v1/chat/completions", json=BODY, headers=HEADERS).status_code == 200


def test_graceful_shutdown_drains_disconnected_attempt(tmp_path):
    model = Model(block=True)
    app = create_app(lambda: model, api_key=TOKEN, output=tmp_path / "runs")
    endpoint = next(route.endpoint for route in app.routes if route.path == "/v1/chat/completions")
    async def receive():
        return {"type":"http.request", "body":json.dumps(BODY).encode(), "more_body":False}
    request = Request({"type":"http", "method":"POST", "path":"/v1/chat/completions",
                       "headers":[(b"content-type", b"application/json")]}, receive)
    async def exercise():
        lifespan = app.router.lifespan_context(app)
        await lifespan.__aenter__()
        attempt = asyncio.create_task(endpoint(request))
        shutdown = None
        try:
            assert await asyncio.to_thread(model.started.wait, 5)
            attempt.cancel()
            with pytest.raises(asyncio.CancelledError):
                await attempt
            shutdown = asyncio.create_task(lifespan.__aexit__(None, None, None))
            await asyncio.sleep(.02)
            assert not shutdown.done()
            model.release.set()
            await asyncio.wait_for(shutdown, 5)
            finished = list((tmp_path / "runs").glob("*-finished.json"))
            assert len(finished) == 1 and json.loads(finished[0].read_text())["http_status"] == 200
        finally:
            model.release.set()
            if shutdown is None:
                await lifespan.__aexit__(None, None, None)
    asyncio.run(exercise())


def test_explicit_cancellation_keeps_worker_locked_and_never_restarts_attempt(tmp_path):
    model = Model(block=True)
    headers = {**HEADERS, "Idempotency-Key": "cancel-me"}
    with TestClient(create_app(lambda: model, api_key=TOKEN, output=tmp_path / "runs")) as client:
        assert client.post("/v1/keyprint/cancel", headers={"Idempotency-Key": "cancel-me"}).status_code == 401
        assert client.post("/v1/keyprint/cancel", headers=HEADERS).status_code == 400
        assert client.post("/v1/keyprint/cancel", headers=headers).status_code == 404
        with ThreadPoolExecutor(max_workers=1) as threads:
            first = threads.submit(client.post, "/v1/chat/completions", json=BODY, headers=headers)
            try:
                assert model.started.wait(5)
                for _ in range(2):
                    cancelled = client.post("/v1/keyprint/cancel", headers=headers)
                    assert cancelled.status_code == 202
                    assert cancelled.json()["state"] == "cancellation_requested"
                assert client.post("/v1/chat/completions", json=BODY, headers=headers).status_code == 409
                assert client.post("/v1/chat/completions", json=BODY, headers=HEADERS).status_code == 503
                assert model.calls == ["hello"]
            finally:
                model.release.set()
            stopped = first.result()
            assert stopped.status_code == 410
        assert client.post("/v1/chat/completions", json=BODY, headers=headers).json() == stopped.json()
        assert client.post("/v1/keyprint/cancel", headers=headers).json() == {
            "state": "terminal", "http_status": 410,
            "message": "Attempt already ended; repeat its original request to recover the result"}
        assert model.calls == ["hello"]
        finished = list((tmp_path / "runs").glob("*-finished.json"))
        assert len(finished) == 1 and json.loads(finished[0].read_text())["http_status"] == 410
        assert client.post("/v1/chat/completions", json=BODY, headers=HEADERS).status_code == 200
        assert model.calls == ["hello", "hello"]


def test_late_cancellation_preserves_completed_success(tmp_path):
    model = Model()
    headers = {**HEADERS, "Idempotency-Key": "already-done"}
    with TestClient(create_app(lambda: model, api_key=TOKEN, output=tmp_path / "runs")) as client:
        original = client.post("/v1/chat/completions", json=BODY, headers=headers)
        assert original.status_code == 200
        assert client.post("/v1/keyprint/cancel", headers=headers).json()["http_status"] == 200
        assert client.post("/v1/chat/completions", json=BODY, headers=headers).json() == original.json()
        assert model.calls == ["hello"]
