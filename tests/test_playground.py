from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import threading

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient
from keyprint import Generation, Inspection
from keyprint.playground import create_playground, inspect_text

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

    def generate(self, prompt, *, max_tokens, condition, output):
        assert threading.get_ident() == self.owner
        self.calls.append(condition)
        self.started.set()
        if self.block:
            assert self.release.wait(10)
        Path(output).mkdir(mode=0o700)
        if self.fail:
            raise RuntimeError("secret internals must stay private")
        return Generation("A seed becomes a tree.", {"completion": "eos", "usage": {"completion_tokens": 6}}, output)

    def inspect(self, text, *, key=None):
        assert threading.get_ident() == self.owner
        return Inspection(1, 16 if key is None else 14, 30,
            {"kind": "literal_diagnostic", "verdict": None})


def client_for(tmp_path, factory=Model, **kwargs):
    return TestClient(create_playground(factory, token=TOKEN, output=tmp_path / "runs", **kwargs),
                      base_url="http://127.0.0.1:8766")


def test_packaged_page_auth_host_and_origin_boundaries(tmp_path):
    with client_for(tmp_path) as client:
        page = client.get("/")
        assert page.status_code == 200 and "Ordinary words." in page.text
        assert TOKEN not in page.text
        assert "frame-ancestors 'none'" in page.headers["Content-Security-Policy"]
        assert client.get("/app.js").status_code == 200
        assert client.get("/app.css").status_code == 200
        assert client.get("/api/session").status_code == 401
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


def test_failed_attempt_stays_failed_and_is_not_silently_retried(tmp_path):
    with client_for(tmp_path, lambda: Model(fail=True)) as client:
        for _ in range(2):
            response = client.post("/api/experiment", json=BODY, headers=HEADERS)
            assert response.status_code == 500 and "secret internals" not in response.text
        assert len(list((tmp_path / "runs").glob("*/ordinary"))) == 1


def test_model_work_cannot_interleave(tmp_path):
    models = []
    def load():
        models.append(Model(block=True))
        return models[0]
    with client_for(tmp_path, load) as client, ThreadPoolExecutor(max_workers=1) as threads:
        first = threads.submit(client.post, "/api/experiment", json=BODY, headers=HEADERS)
        try:
            assert models[0].started.wait(5)
            response = client.post("/api/experiment", json=BODY, headers={**HEADERS, "Idempotency-Key": "other"})
            assert response.status_code == 503
        finally:
            models[0].release.set()
        assert first.result().status_code == 200
        assert models[0].calls == ["ordinary", "marked"]


@pytest.mark.parametrize("body", [{**BODY, "max_tokens": True}, {**BODY, "text": " "},
    {**BODY, "max_tokens": 257}, {**BODY, "text": "x" * 6001}, {**BODY, "extra": "bad"}])
def test_invalid_input_rejected(tmp_path, body):
    with client_for(tmp_path) as client:
        assert client.post("/api/experiment", json=body, headers=HEADERS).status_code == 400


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
