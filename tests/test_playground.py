from concurrent.futures import ThreadPoolExecutor
import asyncio
import json
from pathlib import Path
import threading
import time

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient
from keyprint import Generation, Inspection, KeyprintCancelled
from keyprint.playground import create_playground, inspect_text
from starlette.requests import Request

TOKEN = "playground-fixture-token-" * 2
HEADERS = {"Authorization": "Bearer " + TOKEN, "Idempotency-Key": "experiment-one"}
BODY = {"action": "generate", "text": "Explain a seed.", "max_tokens": 32}


class Model:
    identity = {"profile": "fixture"}

    def __init__(self, *, fail=False, block=False):
        self.owner = threading.get_ident()
        self.calls = []
        self.started, self.release = threading.Event(), threading.Event()
        self.fail, self.block = fail, block

    def generate(self, prompt, *, max_tokens, condition, output, cancel_event=None):
        assert threading.get_ident() == self.owner
        self.calls.append(condition)
        self.started.set()
        if self.block:
            assert self.release.wait(10)
        if cancel_event is not None and cancel_event.is_set():
            raise KeyprintCancelled({"cancellation_requested": True}, Path(output))
        Path(output).mkdir(mode=0o700)
        if self.fail:
            raise RuntimeError("secret internals must stay private")
        return Generation("A seed becomes a tree.", {"completion": "eos", "usage": {"completion_tokens": 6}}, output)

    def inspect(self, text, *, key=None):
        assert threading.get_ident() == self.owner
        return Inspection(1, 16 if key is None else 14, 30,
            {"kind": "literal_diagnostic", "verdict": None})


def client_for(tmp_path, factory=Model, **kwargs):
    class ReadyClient(TestClient):
        def __enter__(self):
            super().__enter__()
            deadline = time.monotonic() + 10
            while self.get("/api/session", headers=HEADERS).json()["model"]["status"] == "loading":
                if time.monotonic() > deadline:
                    raise AssertionError("fixture model did not become ready")
                time.sleep(.01)
            return self
    return ReadyClient(create_playground(factory, token=TOKEN, output=tmp_path / "runs", **kwargs),
                      base_url="http://127.0.0.1:8766")


def test_native_resources_close_on_owning_worker_after_generation(tmp_path):
    closed = []
    class ClosingModel(Model):
        def close(self):
            assert threading.get_ident() == self.owner
            assert self.calls
            closed.append(True)
    with client_for(tmp_path, ClosingModel) as client:
        assert client.post('/api/experiment', json=BODY, headers=HEADERS).status_code == 200
        assert not closed
    assert closed == [True]


def test_loading_serves_page_but_never_queues_or_consumes_a_request(tmp_path):
    entered, release = threading.Event(), threading.Event()
    models, calls = [], []
    def load():
        calls.append(1)
        entered.set()
        assert release.wait(10)
        models.append(Model())
        return models[0]
    app = create_playground(load, token=TOKEN, output=tmp_path / "runs", max_requests=1)
    with TestClient(app, base_url="http://127.0.0.1:8766") as client:
        try:
            assert entered.wait(5)
            assert client.get("/").status_code == 200
            assert client.get("/app.js").status_code == 200
            for _ in range(3):
                session = client.get("/api/session", headers=HEADERS).json()
                assert session["model"]["status"] == "loading" and session["identity"] is None
                assert session["last_attempt"] is None and not session["running"]
                assert client.post("/api/experiment", json=BODY, headers=HEADERS).status_code == 503
            assert not list((tmp_path / "runs").glob("*/request.json"))
        finally:
            release.set()
        deadline = time.monotonic() + 5
        while client.get("/api/session", headers=HEADERS).json()["model"]["status"] == "loading":
            assert time.monotonic() < deadline
            time.sleep(.01)
        response = client.post("/api/experiment", json=BODY, headers=HEADERS)
        assert response.status_code == 200
        assert client.post("/api/experiment", json=BODY, headers=HEADERS).json() == response.json()
        assert len(calls) == 1 and models[0].calls == ["ordinary", "marked"]
        assert json.loads((tmp_path / "runs/startup.json").read_text())["status"] == "ready"


def test_failed_startup_stays_visible_private_and_does_not_reload(tmp_path):
    calls = []
    def load():
        calls.append(1)
        raise ValueError("private model location and failure detail")
    with client_for(tmp_path, load) as client:
        for _ in range(3):
            assert client.get("/").status_code == 200
            response = client.get("/api/session", headers=HEADERS)
            assert response.json()["model"]["status"] == "failed"
            assert response.json()["identity"] is None and response.json()["last_attempt"] is None
            assert "private model location" not in response.text
            result = client.post("/api/experiment", json=BODY, headers=HEADERS)
            assert result.status_code == 503 and "private model location" not in result.text
        assert calls == [1]
        assert not list((tmp_path / "runs").glob("*/request.json"))
        private = json.loads((tmp_path / "runs/startup.json").read_text())
        assert private["error_type"] == "ValueError" and private["detail"] == "private model location and failure detail"


def test_packaged_page_auth_host_and_origin_boundaries(tmp_path):
    with client_for(tmp_path) as client:
        page = client.get("/")
        assert page.status_code == 200 and "Ordinary words." in page.text
        assert TOKEN not in page.text
        assert "frame-ancestors 'none'" in page.headers["Content-Security-Policy"]
        assert client.get("/app.js").status_code == 200
        assert client.get("/reader.js").status_code == 200
        assert client.get("/app.css").status_code == 200
        icon = client.get("/favicon.svg")
        assert icon.status_code == 200 and "image/svg+xml" in icon.headers["content-type"]
        assert client.get("/api/session").status_code == 401
        assert client.get("/api/progress").status_code == 401
        assert client.get("/api/progress", headers=HEADERS).json() == {"active": False}
        assert client.get("/api/session", headers=HEADERS).status_code == 200
        assert client.get("/", headers={"Host": "attacker.example"}).status_code == 403
        assert client.post("/api/experiment", json=BODY,
            headers={**HEADERS, "Origin": "https://attacker.example"}).status_code == 403


def test_real_route_generates_both_conditions_and_replays_without_extra_work(tmp_path):
    with client_for(tmp_path, max_requests=1) as client:
        first = client.post("/api/experiment", json=BODY, headers=HEADERS)
        assert first.status_code == 200
        data = first.json()
        restored = client.get("/api/session", headers=HEADERS).json()
        assert restored["latest"] == data and restored["prompt"] == BODY["text"]
        assert set(data["outputs"]) == {"ordinary", "marked"}
        assert all(value >= 0 for output in data["outputs"].values() for value in output["timing"].values())
        result = data["outputs"]["marked"]["inspection"]
        assert result["series"][-1]["characters"] == len(data["outputs"]["marked"]["text"])
        assert result["series"][-1]["matching"] == result["fraction"]
        assert result["verdict"] is None and not result["calibrated"]
        assert client.post("/api/experiment", json=BODY, headers=HEADERS).json() == data
        assert client.post("/api/experiment", json={**BODY, "text": "changed"}, headers=HEADERS).status_code == 409
        assert client.post("/api/experiment", json=BODY,
            headers={**HEADERS, "Idempotency-Key": "new"}).status_code == 429
        assert TOKEN not in first.text


def test_edit_is_measured_without_generation(tmp_path):
    with client_for(tmp_path) as client:
        response = client.post("/api/experiment", json={"action": "inspect", "text": "Edited."}, headers=HEADERS)
        assert response.status_code == 200 and response.json()["inspection"]["trials"] == 30
        assert not list((tmp_path / "runs").glob("*/marked"))
        session = client.get("/api/session", headers=HEADERS).json()
        assert session["edited"] == {"text": "Edited.", "result": response.json()}


def test_new_pair_clears_previous_edit_but_failed_inspection_retains_it(tmp_path):
    class FailedEdit(Model):
        def inspect(self, text, *, key=None):
            if text.startswith("FAIL"):
                raise RuntimeError("private model failure")
            return super().inspect(text, key=key)

    with client_for(tmp_path, FailedEdit) as client:
        client.post("/api/experiment", json=BODY, headers=HEADERS)
        inspected = client.post("/api/experiment", json={"action": "inspect", "text": "Saved edit."},
                               headers={**HEADERS, "Idempotency-Key": "edit"})
        failed = client.post("/api/experiment", json={"action": "inspect", "text": "FAIL edited text"},
                            headers={**HEADERS, "Idempotency-Key": "failed-edit"})
        assert failed.status_code == 500
        session = client.get("/api/session", headers=HEADERS).json()
        assert session["edited"] == {"text": "Saved edit.", "result": inspected.json()}
        assert session["last_attempt"]["request"]["text"] == "FAIL edited text"
        assert session["last_attempt"]["http_status"] == 500
        assert client.post("/api/experiment", json=BODY,
                           headers={**HEADERS, "Idempotency-Key": "new-pair"}).status_code == 200
        assert client.get("/api/session", headers=HEADERS).json()["edited"] is None


def test_failed_attempt_stays_failed_and_is_not_silently_retried(tmp_path):
    with client_for(tmp_path, lambda: Model(fail=True)) as client:
        for _ in range(2):
            response = client.post("/api/experiment", json=BODY, headers=HEADERS)
            assert response.status_code == 500 and "secret internals" not in response.text
        assert len(list((tmp_path / "runs").glob("*/ordinary"))) == 1
        assert client.get("/api/progress", headers=HEADERS).json() == {"active": False}
        session = client.get("/api/session", headers=HEADERS).json()
        assert session["last_attempt"]["request"] == BODY
        assert session["last_attempt"]["http_status"] == 500


def test_model_work_cannot_interleave(tmp_path):
    models = []
    def load():
        models.append(Model(block=True))
        return models[0]
    with client_for(tmp_path, load) as client, ThreadPoolExecutor(max_workers=1) as threads:
        first = threads.submit(client.post, "/api/experiment", json=BODY, headers=HEADERS)
        try:
            assert models[0].started.wait(5)
            progress = client.get("/api/progress", headers=HEADERS).json()
            assert progress["active"] and progress["stage"] == "generating_ordinary"
            assert progress["seconds"] >= 0
            session = client.get("/api/session", headers=HEADERS).json()
            assert session["running"] and session["latest"] is None
            assert session["last_attempt"]["request"] == BODY
            assert session["last_attempt"]["http_status"] is None
            response = client.post("/api/experiment", json=BODY, headers={**HEADERS, "Idempotency-Key": "other"})
            assert response.status_code == 503
        finally:
            models[0].release.set()
        assert first.result().status_code == 200
        assert models[0].calls == ["ordinary", "marked"]
        assert client.get("/api/progress", headers=HEADERS).json() == {"active": False}
        assert client.get("/api/session", headers=HEADERS).json()["last_attempt"]["request"] == BODY


@pytest.mark.parametrize("body", [{**BODY, "max_tokens": True}, {**BODY, "text": " "},
    {**BODY, "max_tokens": 1025}, {**BODY, "text": "x" * 6001},
    {"action": "inspect", "text": "x" * 16001}, {**BODY, "extra": "bad"}])
def test_invalid_input_rejected(tmp_path, body):
    with client_for(tmp_path) as client:
        assert client.post("/api/experiment", json=body, headers=HEADERS).status_code == 400


def test_longer_budget_is_forwarded_and_long_output_remains_inspectable(tmp_path):
    observed = []

    class LongOutput(Model):
        def generate(self, prompt, *, max_tokens, condition, output, cancel_event=None):
            observed.append(max_tokens)
            result = super().generate(prompt, max_tokens=max_tokens, condition=condition, output=output, cancel_event=cancel_event)
            return Generation("Long output. " * 500, {"completion": "length", "usage": {"completion_tokens": max_tokens}}, result.artifacts)

    with client_for(tmp_path, LongOutput) as client:
        response = client.post("/api/experiment", json={**BODY, "max_tokens": 1024}, headers=HEADERS)
        assert response.status_code == 200
        text = response.json()["outputs"]["marked"]["text"]
        assert len(text) > 6000 and observed == [1024, 1024]
        assert response.json()["max_tokens"] == 1024
        assert client.get("/api/session", headers=HEADERS).json()["latest"]["max_tokens"] == 1024
        assert response.json()["outputs"]["marked"]["completion"] == "length"
        inspected = client.post("/api/experiment", json={"action": "inspect", "text": text},
                               headers={**HEADERS, "Idempotency-Key": "inspect-long"})
        assert inspected.status_code == 200
        assert inspected.json()["inspection"]["series"][-1]["characters"] == len(text)


def test_oversized_body_rejected(tmp_path):
    with client_for(tmp_path) as client:
        assert client.post("/api/experiment", content=b" " * 65537,
            headers={**HEADERS, "Content-Type": "application/json"}).status_code == 413


def test_unavailable_replay_is_not_zero_signal_or_discarded_generation():
    class Unavailable:
        def inspect(self, text, **kwargs):
            raise ValueError("literal boundary unsupported")
    result = inspect_text(Unavailable(), "Some text.", bytes(range(32)))
    assert result["fraction"] is None and result["trials"] is None
    assert all(point["matching"] is None for point in result["series"])
    assert result["report"]["availability"] == "unavailable"


def test_full_text_reports_reuse_final_prefix_including_unavailable():
    class Counter:
        def __init__(self):
            self.calls = []
        def inspect(self, text, *, key=None):
            self.calls.append((text, key))
            return Inspection(1, 15, 30, {"kind": "literal_diagnostic", "call": len(self.calls)})
    model = Counter()
    result = inspect_text(model, "A sufficiently long sentence for sixteen prefixes.", bytes(range(32)))
    assert len(model.calls) == 32
    assert result["report"]["call"] == 31
    assert result["control_report"]["call"] == 32
    assert result["series"][-1]["matching"] == result["fraction"]


def test_cancelled_page_keeps_complete_pair_and_refresh_result(tmp_path):
    models = []
    def load():
        models.append(Model(block=True))
        return models[0]
    app = create_playground(load, token=TOKEN, output=tmp_path / "runs")
    endpoints = {route.path: route.endpoint for route in app.routes}
    def request(key):
        async def receive():
            return {"type": "http.request", "body": json.dumps(BODY).encode(), "more_body": False}
        return Request({"type": "http", "method": "POST", "path": "/api/experiment",
                        "headers": [(b"content-type", b"application/json"), (b"idempotency-key", key.encode())]}, receive)

    async def exercise():
        async with app.router.lifespan_context(app):
            async with asyncio.timeout(5):
                while (await endpoints["/api/session"]())["model"]["status"] == "loading":
                    await asyncio.sleep(.01)
            first = asyncio.create_task(endpoints["/api/experiment"](request("gone")))
            try:
                assert await asyncio.to_thread(models[0].started.wait, 5)
                first.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await first
                waiting = await endpoints["/api/session"]()
                assert waiting["running"] and waiting["last_attempt"]["http_status"] is None
                assert (await endpoints["/api/experiment"](request("new"))).status_code == 503
                models[0].release.set()
                async with asyncio.timeout(5):
                    while True:
                        result = await endpoints["/api/experiment"](request("gone"))
                        if result.status_code != 409:
                            break
                        await asyncio.sleep(.01)
                assert result.status_code == 200
                assert models[0].calls == ["ordinary", "marked"]
                restored = await endpoints["/api/session"]()
                assert restored["latest"] == json.loads(result.body)
                assert not restored["running"] and restored["last_attempt"]["http_status"] == 200
                assert (await endpoints["/api/progress"]()) == {"active": False}
            finally:
                models[0].release.set()
                if not first.done():
                    await first
    asyncio.run(exercise())


@pytest.mark.parametrize("condition", ["ordinary", "marked"])
def test_explicit_stop_preserves_previous_pair_and_edit_and_never_restarts(tmp_path, condition):
    models = []
    class Controlled(Model):
        def __init__(self):
            super().__init__()
            self.held, self.proceed = threading.Event(), threading.Event()
        def generate(self, prompt, **kwargs):
            if prompt == "block" and kwargs["condition"] == condition:
                self.held.set()
                assert self.proceed.wait(10)
            return super().generate(prompt, **kwargs)
    def load():
        models.append(Controlled())
        return models[0]
    with client_for(tmp_path, load) as client, ThreadPoolExecutor(max_workers=1) as threads:
        first = client.post("/api/experiment", json=BODY, headers=HEADERS).json()
        edit = client.post("/api/experiment", json={"action": "inspect", "text": "Saved edit."},
                          headers={**HEADERS, "Idempotency-Key": "saved-edit"}).json()
        headers = {**HEADERS, "Idempotency-Key": "stop-new"}
        assert client.post("/api/cancel", headers={"Idempotency-Key": "stop-new"}).status_code == 401
        assert client.post("/api/cancel", headers=headers).status_code == 404
        body = {**BODY, "text": "block"}
        attempt = threads.submit(client.post, "/api/experiment", json=body, headers=headers)
        try:
            assert models[0].held.wait(5)
            active = client.get("/api/session", headers=HEADERS).json()
            assert active["last_attempt"]["request_id"] == "stop-new"
            for _ in range(2):
                assert client.post("/api/cancel", headers=headers).status_code == 202
            active = client.get("/api/session", headers=HEADERS).json()
            assert active["running"] and active["last_attempt"]["cancellation_requested"]
            progress = client.get("/api/progress", headers=HEADERS).json()
            assert progress["cancellation_requested"] and progress["stage"] == "generating_" + condition
            assert client.post("/api/experiment", json=BODY,
                               headers={**HEADERS, "Idempotency-Key": "busy"}).status_code == 503
        finally:
            models[0].proceed.set()
        stopped = attempt.result()
        assert stopped.status_code == 410
        assert client.post("/api/experiment", json=body, headers=headers).json() == stopped.json()
        restored = client.get("/api/session", headers=HEADERS).json()
        assert not restored["running"] and restored["last_attempt"]["http_status"] == 410
        assert restored["latest"] == first
        assert restored["edited"] == {"text": "Saved edit.", "result": edit}
        assert models[0].calls == ["ordinary", "marked", "ordinary"] + (["marked"] if condition == "marked" else [])
        assert client.post("/api/cancel", headers=headers).json() == {"state": "terminal", "http_status": 410}
        cancelled = json.loads((tmp_path / "runs" / restored["last_attempt"]["run_id"] / "cancelled.json").read_text())
        assert cancelled["completed_conditions"] == (["ordinary"] if condition == "marked" else [])
        assert client.post("/api/experiment", json=BODY,
                           headers={**HEADERS, "Idempotency-Key": "after-stop"}).status_code == 200


def test_stop_inspection_keeps_last_measurement_and_does_not_start_generation(tmp_path):
    models = []
    class Controlled(Model):
        def __init__(self):
            super().__init__()
            self.hold = False
            self.held, self.proceed = threading.Event(), threading.Event()
        def inspect(self, text, **kwargs):
            if self.hold:
                self.held.set()
                assert self.proceed.wait(10)
            return super().inspect(text, **kwargs)
    def load():
        models.append(Controlled())
        return models[0]
    with client_for(tmp_path, load) as client, ThreadPoolExecutor(max_workers=1) as threads:
        saved = client.post("/api/experiment", json={"action": "inspect", "text": "saved"}, headers=HEADERS).json()
        models[0].hold = True
        headers = {**HEADERS, "Idempotency-Key": "inspect-stop"}
        attempt = threads.submit(client.post, "/api/experiment", json={"action": "inspect", "text": "unsaved edits"}, headers=headers)
        try:
            assert models[0].held.wait(5)
            assert client.post("/api/cancel", headers=headers).status_code == 202
        finally:
            models[0].proceed.set()
        assert attempt.result().status_code == 410
        session = client.get("/api/session", headers=HEADERS).json()
        assert session["edited"] == {"text": "saved", "result": saved}
        assert session["last_attempt"]["request"]["text"] == "unsaved edits"
        assert models[0].calls == []
