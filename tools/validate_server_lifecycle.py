"""Real SDK/socket timeout recovery with actual bounded local inference.

A test barrier deliberately holds the worker before inference so timeout and
busy states are deterministic. This validates lifecycle behavior, not latency.
Only public/ is exportable; keys and model journals stay in the private root.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import secrets
import socket
import threading
import time

from keyprint import Keyprint
from keyprint.server import create_app


def main():
    import torch
    import uvicorn
    from openai import OpenAI, APIStatusError, APITimeoutError
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    args.output.mkdir(mode=0o700)
    key = Keyprint.new_key()
    with os.fdopen(os.open(args.output/'owner.key', os.O_WRONLY|os.O_CREAT|os.O_EXCL, 0o600), 'wb') as stream:
        stream.write(key)
    started, release = threading.Event(), threading.Event()
    calls = []
    class ControlledModel:
        def __init__(self):
            self.model = Keyprint.from_transformers(args.model, key=key)
        def __getattr__(self, name):
            return getattr(self.model, name)
        def generate(self, prompt, **kwargs):
            calls.append(prompt)
            started.set()
            if len(calls) == 1 and not release.wait(30):
                raise RuntimeError('test barrier was not released')
            return self.model.generate(prompt, **kwargs)
    token = secrets.token_hex(32)
    server = uvicorn.Server(uvicorn.Config(create_app(ControlledModel, api_key=token,
        output=args.output/'http'), log_level='warning'))
    sock = socket.socket(); sock.bind(('127.0.0.1', 0))
    port = sock.getsockname()[1]
    thread = threading.Thread(target=server.run, kwargs={'sockets':[sock]}, daemon=True)
    thread.start()
    report = {'scope':'Real local inference with deliberate pre-inference test barrier; not a latency measurement',
              'hosted_provider_calls':False, 'checks':{}, 'status':'running'}
    failed = False
    try:
        deadline = time.monotonic()+120
        while not server.started:
            if not thread.is_alive() or time.monotonic() > deadline:
                raise RuntimeError('server startup failed')
            time.sleep(.05)
        with OpenAI(base_url=f'http://127.0.0.1:{port}/v1', api_key=token, max_retries=0, timeout=120) as client:
            params = dict(model='keyprint', messages=[{'role':'user','content':'Explain how rain forms in two sentences.'}],
                          max_completion_tokens=96, extra_headers={'Idempotency-Key':'timeout-recovery'})
            def first_request():
                try:
                    client.with_options(timeout=.3).chat.completions.create(**params)
                except APITimeoutError:
                    return True
                return False
            def expected_error(code, **overrides):
                try:
                    client.chat.completions.create(**{**params, **overrides})
                except APIStatusError as exc:
                    assert exc.status_code == code, (exc.status_code, code)
                    return
                raise AssertionError(f'expected HTTP {code}')
            with ThreadPoolExecutor(max_workers=1) as executor:
                attempt = executor.submit(first_request)
                assert started.wait(15), 'request never reached model worker'
                assert attempt.result(timeout=5), 'expected client timeout'
            report['checks']['actual_client_timeout'] = True
            expected_error(409)  # Same accepted work is still running.
            expected_error(409, max_completion_tokens=32)  # ID conflict.
            expected_error(503, extra_headers={'Idempotency-Key':'while-busy'})
            assert len(calls) == 1
            report['checks']['running_replay_conflict_and_busy_rejected'] = True
            release.set()
            deadline = time.monotonic()+120
            while True:
                try:
                    recovered = client.chat.completions.create(**params)
                    break
                except APIStatusError as exc:
                    assert exc.status_code == 409
                    if time.monotonic() > deadline:
                        raise RuntimeError('accepted attempt did not become replayable')
                    time.sleep(.1)
            replay = client.chat.completions.create(**params)
            assert replay.model_dump() == recovered.model_dump()
            assert recovered.choices[0].message.content.strip()
            assert len(calls) == 1
            reports = list((args.output/'http').glob('chatcmpl-*/report.json'))
            assert len(reports) == 1
            saved = json.loads(reports[0].read_text())
            assert saved['text'] == recovered.choices[0].message.content
            report['checks']['completed_result_recovered_without_second_generation'] = True
            report['text'] = saved['text']
            report['finish_reason'] = recovered.choices[0].finish_reason
            report['usage'] = recovered.usage.model_dump()
            # Busy rejection must not consume that request ID or a model attempt.
            next_result = client.chat.completions.create(**{**params,
                'max_completion_tokens':32, 'extra_headers':{'Idempotency-Key':'while-busy'}})
            assert next_result.id != recovered.id and len(calls) == 2
            report['checks']['worker_available_after_recovery'] = True
            report['status'] = 'pass'
    except Exception as exc:
        report.update(status='failed', error_type=type(exc).__name__)
        failed = True
    finally:
        release.set()
        server.should_exit = True
        thread.join(timeout=180)
        sock.close()
        report['model_attempts'] = len(calls)
        report['checks']['graceful_shutdown'] = not thread.is_alive()
        if thread.is_alive():
            report['status'] = 'failed'
            failed = True
        public = args.output/'public'; public.mkdir()
        (public/'lifecycle.json').write_text(json.dumps(report, indent=2, ensure_ascii=False))
        print(json.dumps(report, indent=2, ensure_ascii=False))
        if os.environ.get('GITHUB_STEP_SUMMARY'):
            with open(os.environ['GITHUB_STEP_SUMMARY'], 'a') as stream:
                stream.write('\n## Actual inference lifecycle\n\nTimeout recovery: ' + report['status'] +
                    '. Tests use a deliberate pre-inference barrier; no performance claim. ' +
                    'See lifecycle.json in the artifact for text, attempts and checks.\n')
    return int(failed)


if __name__ == '__main__':
    raise SystemExit(main())
