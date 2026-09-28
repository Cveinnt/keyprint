"""Actual PydanticAI text/NativeOutput requests against a local MLX worker."""
import argparse
import asyncio
from dataclasses import asdict
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import secrets
import socket
import threading
import time

os.environ['OTEL_SDK_DISABLED'] = 'true'
os.environ['LOGFIRE_SEND_TO_LOGFIRE'] = 'false'
os.environ['PYDANTIC_AI_NO_BANNER'] = '1'
from openai import AsyncOpenAI
from pydantic import BaseModel, ConfigDict
from pydantic_ai import Agent
from pydantic_ai.exceptions import ModelHTTPError, UserError
import uvicorn
import keyprint
from keyprint import Keyprint
from keyprint.server import create_app

recipe_path = Path(__file__).parents[1] / 'examples/pydantic_ai_local.py'
spec = importlib.util.spec_from_file_location('pydantic_ai_local_recipe', recipe_path)
recipe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recipe)

CASES = [
    ('english', 'Explain in one sentence why a seed needs water.'),
    ('french', 'En français, écris une phrase demandant à Maya de garder la sauvegarde jusqu’à la vérification de la restauration.'),
    ('spanish', 'En español, escribe una frase pidiendo a Maya que espere mi aprobación antes de publicar el documento.'),
    ('structured', 'Return JSON using only these facts: Maya reviews the draft by Tuesday 09:30. Vincent approves publication. Publication cannot proceed without approval. Use fields reviewer, deadline, approver, publish_without_approval. Preserve the names and exact deadline.'),
]


class Approval(BaseModel):
    model_config = ConfigDict(extra='forbid')
    reviewer: str
    deadline: str
    approver: str
    publish_without_approval: bool


def save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--wheel', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.output
    root.mkdir(mode=0o700)
    public = root / 'public'; public.mkdir()
    versions = {n: importlib.metadata.version(n) for n in
                ('keyprint', 'pydantic-ai-slim', 'openai', 'httpx', 'pydantic', 'mlx', 'mlx-lm', 'llguidance')}
    key, token = Keyprint.new_key(), secrets.token_hex(32)
    (root / 'owner.key').write_bytes(key); (root / 'owner.key').chmod(0o600)
    save(public / 'plan.json', {'cases': CASES, 'max_tokens': 128, 'backend': 'MLX reference',
        'revision': args.model.name, 'versions': versions, 'approval_schema': Approval.model_json_schema(),
        'wheel_sha256': hashlib.sha256(args.wheel.read_bytes()).hexdigest(),
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'recipe_sha256': hashlib.sha256(recipe_path.read_bytes()).hexdigest(),
        'installed_package': str(Path(keyprint.__file__).resolve()),
        'scope': 'One-turn text and explicit native JSON; replay, request conflict and rejection checks. No tools, system prompts, multi-turn, streaming, detection or quality acceptance.',
        'hosted_calls': False, 'external_tracing': False, 'no_automatic_retries': True})
    report = {'status': 'running', 'ordinary': [], 'marked': [], 'rejections': {}, 'quality_acceptance': False}
    calls = []

    def loader():
        wm = Keyprint.from_mlx(args.model, key=key)
        report['identity'] = wm.identity
        for name, prompt in CASES:
            output = wm.generate(prompt, max_tokens=128, condition='ordinary', output=root / ('ordinary-' + name),
                                 **({'json_schema': Approval.model_json_schema()} if name == 'structured' else {}))
            report['ordinary'].append({'id': name, 'prompt': prompt, 'text': output.text,
                                      'completion': output.report['payload']['completion'], 'usage': output.report['usage']})
            save(public / 'results.json', report)
        generate = wm.generate
        def counted(*a, **kw):
            calls.append(kw.get('condition', 'marked'))
            return generate(*a, **kw)
        wm.generate = counted
        return wm

    sock = socket.socket(); sock.bind(('127.0.0.1', 0))
    base_url = f'http://127.0.0.1:{sock.getsockname()[1]}/v1'
    server = uvicorn.Server(uvicorn.Config(create_app(loader, api_key=token, output=root / 'http'), log_level='error'))
    thread = threading.Thread(target=server.run, kwargs={'sockets': [sock]}, daemon=True)
    thread.start()

    def record(name, prompt, first, replay, mode):
        output = first.output.model_dump() if isinstance(first.output, BaseModel) else first.output
        repeated = replay.output.model_dump() if isinstance(replay.output, BaseModel) else replay.output
        if output != repeated or asdict(first.usage) != asdict(replay.usage):
            raise AssertionError('Pydantic output or usage changed during replay')
        if len(calls) != len(report['marked']) + 1:
            raise AssertionError('Replay triggered extra model work')
        report['marked'].append({'id': name, 'prompt': prompt, 'output': output, 'usage': asdict(first.usage),
                                 'replay_exact': True, 'mode': mode})
        save(public / 'results.json', report)
        print(name + ' completed with exact replay', flush=True)

    try:
        deadline = time.monotonic() + 300
        while not server.started:
            if not thread.is_alive() or time.monotonic() > deadline:
                raise RuntimeError('Local server startup failed')
            time.sleep(.1)
        loop = asyncio.new_event_loop(); asyncio.set_event_loop(loop)
        client = AsyncOpenAI(base_url=base_url, api_key=token, max_retries=0, timeout=180)
        try:
            agent = recipe.local_agent(client)
            name, prompt = CASES[0]
            settings = {'max_tokens': 128, 'extra_headers': {'Idempotency-Key': 'pydantic-' + name}}
            first = agent.run_sync(prompt, model_settings=settings)
            replay = agent.run_sync(prompt, model_settings=settings)
            record(name, prompt, first, replay, 'run_sync')
        finally:
            loop.run_until_complete(client.close()); loop.close(); asyncio.set_event_loop(None)

        async def remaining():
            async with AsyncOpenAI(base_url=base_url, api_key=token, max_retries=0, timeout=180) as client:
                text_agent = recipe.local_agent(client)
                history = None
                for name, prompt in CASES[1:]:
                    agent = recipe.local_agent(client, output_type=Approval) if name == 'structured' else text_agent
                    settings = {'max_tokens': 128, 'extra_headers': {'Idempotency-Key': 'pydantic-' + name}}
                    first = await agent.run(prompt, model_settings=settings)
                    replay = await agent.run(prompt, model_settings=settings)
                    record(name, prompt, first, replay, 'run')
                    if name == 'structured':
                        expected = Approval(reviewer='Maya', deadline='Tuesday 09:30', approver='Vincent', publish_without_approval=False)
                        report['structured_values_match'] = first.output == expected
                    else:
                        history = first.all_messages()
                for name, target, kwargs in [
                    ('history', text_agent, {'message_history': history}),
                    ('instructions', Agent(text_agent.model, instructions='Answer briefly.', retries=0), {}),
                    ('conflict', text_agent, {}),
                ]:
                    request_id = 'pydantic-english' if name == 'conflict' else 'reject-' + name
                    try:
                        await target.run('A different request.', model_settings={'max_tokens': 128,
                            'extra_headers': {'Idempotency-Key': request_id}}, **kwargs)
                    except ModelHTTPError as error:
                        expected_status = 409 if name == 'conflict' else 400
                        if error.status_code != expected_status:
                            raise
                        report['rejections'][name] = expected_status
                    else:
                        raise AssertionError(name + ' unexpectedly accepted')
                try:
                    async with text_agent.run_stream('Hello', model_settings={'max_tokens': 32,
                        'extra_headers': {'Idempotency-Key': 'reject-stream'}}):
                        raise AssertionError('Streaming unexpectedly accepted')
                except ModelHTTPError as error:
                    if error.status_code != 400:
                        raise
                    report['rejections']['stream'] = 400
                try:
                    await Agent(text_agent.model, output_type=Approval, retries=0).run('Return approval.')
                except UserError:
                    report['rejections']['implicit_tool_output'] = 'client profile rejection'
                else:
                    raise AssertionError('Tool output unexpectedly accepted')
        asyncio.run(remaining())
        journals = list((root / 'http').rglob('journal.jsonl'))
        if len(calls) != 4 or len(journals) != 4:
            raise AssertionError('Replay or rejection consumed extra model requests')
        saved = []
        for path in (root / 'http').rglob('report.json'):
            value = json.loads(path.read_text())['report']
            saved.append((value['rendered_carriers']['visible_text'], value['usage']['completion_tokens']))
        for row in report['marked']:
            matches = [(text, tokens) for text, tokens in saved if tokens == row['usage']['output_tokens']
                       and (json.loads(text) == row['output'] if isinstance(row['output'], dict) else text == row['output'])]
            if len(matches) != 1:
                raise AssertionError('Returned text or usage does not match a unique receipt')
        report.update(status='pass', actual_marked_requests=4, journals=4, exact_text_and_usage_match_private_receipts=True)
    except BaseException as error:
        report.update(status='failed', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        server.should_exit = True
        thread.join(60); sock.close()
        report['server_stopped'] = not thread.is_alive()
        save(public / 'results.json', report)
        if thread.is_alive():
            raise RuntimeError('Local server did not stop')


if __name__ == '__main__':
    main()
