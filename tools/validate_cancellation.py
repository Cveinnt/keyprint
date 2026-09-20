"""Actual local inference cancellation, terminal replay and worker reuse.

A barrier after a real forward and at least one committed token makes the
cancellation race reproducible. No model outputs or sampling draws are replaced.
This is a lifecycle test, not an inference-latency or quality benchmark.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import secrets
import socket
import threading
import time

from keyprint import Keyprint
from keyprint.server import create_app


def journal_counts(path):
    previous = "0" * 64
    committed = 0
    for index, line in enumerate(path.read_bytes().splitlines(keepends=True)):
        row = json.loads(line)
        assert row["sequence"] == index and row["previous_sha256"] == previous
        previous = hashlib.sha256(line).hexdigest()
        event = row["event"]
        committed += event.get("phase") == "committed" or event.get("kind") == "committed_step"
    return committed


def main():
    import httpx
    import uvicorn
    from openai import OpenAI, APIStatusError
    from anthropic import Anthropic, APIStatusError as AnthropicStatusError
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["transformers", "mlx"], required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execution", choices=["reference", "experimental-fast", "experimental-native"], default="reference")
    parser.add_argument("--protocol", choices=["openai", "anthropic"], default="openai")
    args = parser.parse_args()
    if args.backend != "mlx" and args.execution != "reference":
        parser.error("experimental execution requires the MLX backend")
    if args.backend == "transformers":
        import torch
        torch.set_num_threads(1)
    args.output.mkdir(mode=0o700)
    key = Keyprint.new_key()
    with os.fdopen(os.open(args.output / "owner.key", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb") as stream:
        stream.write(key)
    blocked, release = threading.Event(), threading.Event()
    attempts, forwards = [], []
    state = {"output": None, "held": False, "first_forward_count": None}

    class ObservedModel:
        def __init__(self, original):
            self.original = original

        def __getattr__(self, name):
            return getattr(self.original, name)

        def __call__(self, *args, **kwargs):
            result = self.original(*args, **kwargs)
            forwards.append(1)
            path = state["output"] / "journal.jsonl"
            if len(attempts) == 1 and not state["held"] and journal_counts(path) >= 1:
                state["held"] = True
                state["first_forward_count"] = len(forwards)
                blocked.set()
                if not release.wait(30):
                    raise RuntimeError("cancellation test barrier was not released")
            return result

    class ControlledModel:
        def __init__(self):
            loader = Keyprint.from_transformers if args.backend == "transformers" else Keyprint.from_mlx
            self.model = loader(args.model, key=key, **({"execution": args.execution} if args.backend == "mlx" else {}))
            state["identity"] = self.model.identity
            self.model._backend.model = ObservedModel(self.model._backend.model)

        def generate(self, prompt, **kwargs):
            attempts.append(prompt)
            state["output"] = Path(kwargs["output"])
            return self.model.generate(prompt, **kwargs)

    token = secrets.token_hex(32)
    server = uvicorn.Server(uvicorn.Config(create_app(ControlledModel, api_key=token,
        output=args.output / "http"), log_level="warning"))
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    base = f"http://127.0.0.1:{sock.getsockname()[1]}/v1"
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    report = {"status": "running", "backend": args.backend, "execution": args.execution, "protocol": args.protocol, "checks": {},
              "scope": "Actual local inference with a deliberate post-forward barrier; no quality, preemption or latency claim",
              "hosted_provider_calls": False}
    failed = False
    try:
        deadline = time.monotonic() + 120
        while not server.started:
            if not thread.is_alive() or time.monotonic() > deadline:
                raise RuntimeError("server did not start")
            time.sleep(.05)
        client_class = OpenAI if args.protocol == 'openai' else Anthropic
        client_base = base if args.protocol == 'openai' else base.removesuffix('/v1')
        cap_field = 'max_completion_tokens' if args.protocol == 'openai' else 'max_tokens'
        with client_class(base_url=client_base, api_key=token, max_retries=0, timeout=180) as client:
            create = client.chat.completions.create if args.protocol == 'openai' else client.messages.create
            params = {"model": "keyprint", "messages": [{"role": "user", "content":
                      "Explain how rain forms in a detailed paragraph of about 150 words."}],
                      cap_field: 128, "extra_headers": {"Idempotency-Key": "cancel-real"}}

            def expect_error(code, values):
                try:
                    create(**values)
                except (APIStatusError, AnthropicStatusError) as exc:
                    assert exc.status_code == code, (exc.status_code, code)
                    return exc.response.json()
                raise AssertionError(f"expected HTTP {code}")

            headers = {"Authorization": "Bearer " + token, "Idempotency-Key": "cancel-real"}
            with ThreadPoolExecutor(max_workers=1) as pool:
                first = pool.submit(expect_error, 410, params)
                try:
                    assert blocked.wait(90), "generation did not reach the partial-token test barrier"
                    committed_before = journal_counts(state["output"] / "journal.jsonl")
                    assert committed_before >= 1
                    assert httpx.post(base + "/keyprint/cancel", headers={"Idempotency-Key": "cancel-real"}).status_code == 401
                    for _ in range(2):
                        response = httpx.post(base + "/keyprint/cancel", headers=headers, timeout=10)
                        assert response.status_code == 202
                        assert response.json()["state"] == "cancellation_requested"
                    expect_error(409, params)
                    expect_error(503, {**params, "extra_headers": {"Idempotency-Key": "while-cancelling"}})
                    assert len(attempts) == 1
                    report["checks"]["cancellation_requested_after_real_partial_generation"] = True
                    report["checks"]["worker_remains_locked_until_boundary"] = True
                finally:
                    release.set()
                stopped = first.result(timeout=60)
            assert len(forwards) == state["first_forward_count"]
            assert expect_error(410, params) == stopped
            assert len(attempts) == 1
            folder = state["output"]
            saved = json.loads((folder / "report.json").read_text())
            saved = saved.get("report", saved)
            assert saved["kind"] == "error" and saved["cancellation_requested"] is True
            assert journal_counts(folder / "journal.jsonl") == committed_before
            assert saved["usage"]["completion_tokens"] == committed_before
            report["checks"]["no_additional_forward_or_token_commit_after_cancellation"] = True
            report["checks"]["cancelled_attempt_replays_without_restart"] = True
            report["cancelled_usage"] = saved["usage"]
            report["cancelled_forward_calls"] = state["first_forward_count"]
            terminal = httpx.post(base + "/keyprint/cancel", headers=headers)
            assert terminal.status_code == 200 and terminal.json()["http_status"] == 410
            next_params = {**params, cap_field: 32,
                           "extra_headers": {"Idempotency-Key": "while-cancelling"}}
            completed = create(**next_params)
            text = completed.choices[0].message.content if args.protocol == 'openai' else completed.content[0].text
            finish_reason = completed.choices[0].finish_reason if args.protocol == 'openai' else completed.stop_reason
            assert len(attempts) == 2 and text.strip()
            assert create(**next_params).model_dump() == completed.model_dump()
            assert len(attempts) == 2
            late = httpx.post(base + "/keyprint/cancel", headers={**headers, "Idempotency-Key": "while-cancelling"})
            assert late.status_code == 200 and late.json()["http_status"] == 200
            report["checks"]["new_attempt_and_terminal_replay_succeed"] = True
            report["checks"]["late_cancel_preserves_success"] = True
            report["next_text"] = text
            report["next_finish_reason"] = finish_reason
            report["next_usage"] = completed.usage.model_dump()
            report["status"] = "pass"
    except Exception as exc:
        report.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        failed = True
    finally:
        release.set()
        server.should_exit = True
        thread.join(timeout=180)
        sock.close()
        report["model_attempts"] = len(attempts)
        report["identity"] = state.get("identity")
        report["checks"]["graceful_shutdown"] = not thread.is_alive()
        if thread.is_alive():
            report["status"] = "failed"
            failed = True
        public = args.output / "public"
        public.mkdir()
        (public / "cancellation.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
        print(json.dumps(report, indent=2, ensure_ascii=False))
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
