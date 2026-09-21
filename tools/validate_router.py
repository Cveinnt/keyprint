"""Real local LiteLLM routing and recovery. Only public/ is shareable.

No mocked completions. A barrier before one model call makes timeout/busy checks
repeatable; this is an integration test, not a performance measurement.
"""
import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import secrets
import socket
import threading
import time

os.environ['LITELLM_LOCAL_MODEL_COST_MAP'] = 'True'
os.environ['LANGSMITH_TRACING'] = 'false'
os.environ['LANGCHAIN_TRACING_V2'] = 'false'
import litellm
import uvicorn
from keyprint import Keyprint
from keyprint.server import create_app

litellm.telemetry = False


def main():
    if not __debug__:
        raise RuntimeError('run without -O; assertions validate real inference')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.output
    root.mkdir(mode=0o700)
    public = root / 'public'; public.mkdir()
    with args.model.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    plan = {'model_sha256': digest, 'backend': 'llama-cpp', 'threads_per_worker': 2,
            'context_size': 2048, 'max_tokens': 64, 'workers': 2,
            'cases': ['Explain in one sentence why a seed needs water.',
                      'En français, demande à Maya de garder la sauvegarde jusqu’à la vérification de la restauration.'],
            'versions': {name: importlib.metadata.version(name) for name in
                         ['litellm', 'openai', 'llama-cpp-python']},
            'scope': 'Explicit routes, sync/async text and same-worker replay; not production failover',
            'num_retries': 0, 'max_fallbacks': 0, 'cache_responses': False}
    (public / 'plan.json').write_text(json.dumps(plan, indent=2, ensure_ascii=False))
    report = {'status': 'running', 'ordinary': [], 'marked': [], 'checks': {},
              'quality_acceptance': False, 'hosted_calls': False}
    key = Keyprint.new_key()
    with os.fdopen(os.open(root/'owner.key', os.O_WRONLY|os.O_CREAT|os.O_EXCL, 0o600), 'wb') as f:
        f.write(key)
    calls = [[], []]; observed = [[], []]
    entered, release = threading.Event(), threading.Event()
    servers = []; routes = []; router = None
    def save():
        (public/'results.json').write_text(json.dumps(report, indent=2, ensure_ascii=False))
    try:
        for index in range(2):
            token = secrets.token_hex(32)
            def loader(index=index):
                model = Keyprint.from_llama_cpp(args.model, key=key, context_size=2048, threads=2)
                original = model.generate
                ordinary = original(plan['cases'][index], max_tokens=64, condition='ordinary',
                                    output=root/f'ordinary-{index}')
                report['ordinary'].append({'route': index, 'text': ordinary.text, 'usage': ordinary.report['usage']})
                save()
                def generate(prompt, **kwargs):
                    calls[index].append(prompt)
                    if prompt == 'Explain how rain forms in one sentence.':
                        entered.set()
                        if not release.wait(30):
                            raise RuntimeError('barrier not released')
                    return original(prompt, **kwargs)
                model.generate = generate
                return model
            app = create_app(loader, api_key=token, output=root/f'http-{index}')
            @app.middleware('http')
            async def record(request, call_next, index=index):
                if request.method == 'POST':
                    observed[index].append({'path': request.url.path,
                                            'idempotency': request.headers.get('idempotency-key')})
                return await call_next(request)
            sock = socket.socket(); sock.bind(('127.0.0.1', 0))
            server = uvicorn.Server(uvicorn.Config(app, log_level='error'))
            thread = threading.Thread(target=server.run, kwargs={'sockets': [sock]}, daemon=True)
            servers.append((server, thread, sock)); thread.start()
            deadline = time.monotonic()+240
            while not server.started:
                if not thread.is_alive() or time.monotonic() > deadline:
                    raise RuntimeError('local worker startup failed')
                time.sleep(.1)
            routes.append({'model_name': f'keyprint-{index}', 'litellm_params': {
                'model': 'openai/keyprint', 'api_base': f'http://127.0.0.1:{sock.getsockname()[1]}/v1',
                'api_key': token, 'max_retries': 0}})
        router = litellm.Router(model_list=routes, num_retries=0, max_fallbacks=0,
                               fallbacks=[], cache_responses=False, timeout=180,
                               disable_cooldowns=True)
        def params(index, prompt, identifier):
            return dict(model=f'keyprint-{index}', messages=[{'role':'user', 'content':prompt}],
                        max_tokens=64, extra_headers={'Idempotency-Key':identifier})
        for index in range(2):
            kwargs = params(index, plan['cases'][index], f'route-{index}')
            if index:
                async def invoke():
                    first = await router.acompletion(**kwargs)
                    replay = await router.acompletion(**kwargs)
                    return first, replay
                first, replay = asyncio.run(invoke())
            else:
                first, replay = router.completion(**kwargs), router.completion(**kwargs)
            assert first.choices[0].message.content == replay.choices[0].message.content
            assert first.usage == replay.usage
            assert len(calls[index]) == 1
            report['marked'].append({'route':index, 'mode':'async' if index else 'sync',
                'prompt':plan['cases'][index], 'text':first.choices[0].message.content,
                'usage':first.usage.model_dump(), 'replay_exact':True})
            save(); print('route', index, 'passed', flush=True)
        assert [len(x) for x in calls] == [1, 1]
        report['checks']['explicit_route_isolation_and_header_forwarding'] = True
        for index in range(2):
            assert [x['idempotency'] for x in observed[index]] == [f'route-{index}']*2
        # Invalid input may not be dropped, retried, or sent to another route.
        before = [len(x) for x in calls]
        try:
            router.completion(**params(0, 'Hello', 'invalid-tools'), tools=[{
                'type':'function', 'function':{'name':'noop','parameters':{'type':'object'}}}])
        except litellm.BadRequestError as exc:
            assert exc.status_code == 400
        else:
            raise AssertionError('tools accepted')
        assert [len(x) for x in calls] == before
        report['checks']['unsupported_features_do_not_fallback'] = True
        timed = params(0, 'Explain how rain forms in one sentence.', 'timed-attempt')
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(router.completion, **timed, timeout=.3)
            assert entered.wait(10)
            try:
                future.result(timeout=10)
            except litellm.Timeout:
                pass
            else:
                raise AssertionError('expected real client timeout')
        assert [len(x) for x in calls] == [2, 1]
        for kwargs, code in [(timed, 409), (params(0, 'Hello', 'busy'), 503)]:
            try:
                router.completion(**kwargs)
            except (litellm.APIError, litellm.ServiceUnavailableError) as exc:
                assert exc.status_code == code, (type(exc).__name__, exc.status_code)
            else:
                raise AssertionError('expected running/busy rejection')
        assert [len(x) for x in calls] == [2, 1]
        release.set()
        deadline = time.monotonic()+180
        while True:
            try:
                recovered = router.completion(**timed)
                break
            except (litellm.APIError, litellm.ServiceUnavailableError) as exc:
                assert exc.status_code == 409
                if time.monotonic()>deadline: raise RuntimeError('recovery deadline')
                time.sleep(.5)
        report['marked'].append({'route':0,'mode':'timeout-recovery','prompt':timed['messages'][0]['content'],
            'text':recovered.choices[0].message.content,'usage':recovered.usage.model_dump()})
        assert [len(x) for x in calls] == [2, 1]
        report['checks']['timeout_busy_recovery_without_retry_or_fallback'] = True
        for index in range(2):
            saved=[json.loads(p.read_text()) for p in (root/f'http-{index}').rglob('report.json')]
            actual=[r for r in report['marked'] if r['route']==index]
            assert len(saved)==len(actual)==len(calls[index])
            for row in actual:
                assert any(r['text']==row['text'] and all(r['usage'][k]==row['usage'][k] for k in ('prompt_tokens','completion_tokens','total_tokens')) for r in saved)
        report['checks']['texts_and_usage_match_private_reports'] = True
        report['status']='pass'
    except BaseException as exc:
        report.update(status='failed', error_type=type(exc).__name__)
        raise
    finally:
        release.set()
        if router: router.reset()
        for server, thread, sock in servers:
            server.should_exit=True; thread.join(180); sock.close()
        report['checks']['graceful_shutdown']=all(not t.is_alive() for _,t,_ in servers)
        report['model_calls_by_route']=[len(x) for x in calls]
        report['http_requests_by_route']=[len(x) for x in observed]
        if not report['checks']['graceful_shutdown']:report['status']='failed'
        save()
    return 0 if report['status']=='pass' else 1


if __name__ == '__main__':
    raise SystemExit(main())
