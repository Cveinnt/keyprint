"""Real disk/restart boundaries with deterministic generation fixtures."""
import subprocess
import sqlite3
import sys

import pytest
pytest.importorskip('fastapi')
from fastapi.testclient import TestClient

from keyprint.replay import ReplayStore
from keyprint.server import create_app
from test_server import Model, TOKEN, HEADERS, BODY


@pytest.mark.parametrize('protocol', ['openai', 'anthropic', 'ollama-chat', 'ollama-generate'])
@pytest.mark.parametrize('failed', [False, True])
def test_restart_replays_exact_terminal_response_without_generation(tmp_path, protocol, failed):
    path, body = '/v1/chat/completions', BODY
    headers = {**HEADERS, 'Idempotency-Key': 'durable', 'anthropic-version': '2023-06-01'}
    if protocol == 'anthropic':
        path, body = '/v1/messages', {**BODY, 'max_tokens': 32}
    elif protocol.startswith('ollama'):
        path = '/api/chat' if protocol == 'ollama-chat' else '/api/generate'
        body = {'model': 'keyprint', 'stream': False, 'options': {'num_predict': 32},
                **({'messages': BODY['messages']} if protocol == 'ollama-chat' else {'prompt': 'hello'})}
    first_model = Model(fail=failed)
    with TestClient(create_app(lambda: first_model, api_key=TOKEN, output=tmp_path/'runs', max_requests=1)) as client:
        first = client.post(path, json=body, headers=headers)
        assert first.status_code == (500 if failed else 200)
    second_model = Model()
    with TestClient(create_app(lambda: second_model, api_key=TOKEN, output=tmp_path/'runs', max_requests=1)) as client:
        second = client.post(path, json=body, headers=headers)
        assert (second.status_code, second.content, dict(second.headers)) == (first.status_code, first.content, dict(first.headers))
        assert client.post(path, json={**body, 'model': 'keyprint',
            **({'prompt': 'changed'} if protocol == 'ollama-generate' else {'messages': [{'role': 'user', 'content': 'changed'}]})}, headers=headers).status_code == 409
        assert client.post(path, json=body, headers={**headers, 'Idempotency-Key': 'new'}).status_code == 429
        assert second_model.calls == []
    assert first_model.calls == ['hello']


def test_interrupted_reservation_is_terminal_after_restart(tmp_path, monkeypatch):
    original = ReplayStore.finish
    def disk_failure(*args):
        raise OSError('private failure')
    model = Model()
    headers = {**HEADERS, 'Idempotency-Key': 'interrupted'}
    with TestClient(create_app(lambda: model, api_key=TOKEN, output=tmp_path/'runs')) as client:
        monkeypatch.setattr(ReplayStore, 'finish', disk_failure)
        assert client.post('/v1/chat/completions', json=BODY, headers=headers).status_code == 500
    monkeypatch.setattr(ReplayStore, 'finish', original)
    recovered = Model()
    with TestClient(create_app(lambda: recovered, api_key=TOKEN, output=tmp_path/'runs')) as client:
        result = client.post('/v1/chat/completions', json=BODY, headers=headers)
        assert result.status_code == 410 and result.headers['x-should-retry'] == 'false'
        assert 'will not restart' in result.text and recovered.calls == []
        assert client.post('/v1/keyprint/cancel', headers=headers).json()['state'] == 'terminal'


@pytest.mark.parametrize('changed', ['model', 'key', 'auth'])
def test_reusing_directory_with_different_binding_fails_closed(tmp_path, changed):
    with TestClient(create_app(Model, api_key=TOKEN, output=tmp_path/'runs')):
        pass
    model = Model()
    if changed == 'model': model.identity = {'fixture': 'different-model'}
    if changed == 'key': model._key = bytes([1])*32
    auth = 'changed-token-'*4 if changed == 'auth' else TOKEN
    with pytest.raises(ValueError, match='different model'):
        with TestClient(create_app(lambda: model, api_key=auth, output=tmp_path/'runs')):
            pytest.fail('changed binding accepted')
    assert model.calls == []
    # Failed startup must release the output lock.
    with TestClient(create_app(Model, api_key=TOKEN, output=tmp_path/'runs')):
        pass


def test_directory_cannot_be_owned_by_two_servers(tmp_path):
    with TestClient(create_app(Model, api_key=TOKEN, output=tmp_path/'runs')):
        with pytest.raises(BlockingIOError):
            with TestClient(create_app(lambda: pytest.fail('second model loaded'), api_key=TOKEN, output=tmp_path/'runs')):
                pytest.fail('second owner accepted')


def test_reservation_failure_prevents_model_work(tmp_path, monkeypatch):
    model = Model()
    with TestClient(create_app(lambda: model, api_key=TOKEN, output=tmp_path/'runs')) as client:
        def fail(*args): raise OSError('private disk failure')
        monkeypatch.setattr(ReplayStore, 'reserve', fail)
        response = client.post('/v1/chat/completions', json=BODY, headers=HEADERS)
        assert response.status_code == 503 and model.calls == []
        assert 'private disk failure' not in response.text


def test_journal_read_failure_prevents_work_and_disables_client_retry(tmp_path, monkeypatch):
    model = Model()
    with TestClient(create_app(lambda: model, api_key=TOKEN, output=tmp_path/'runs')) as client:
        def fail(*args): raise OSError('private read failure')
        monkeypatch.setattr(ReplayStore, 'get', fail)
        for path in ('/v1/chat/completions', '/v1/keyprint/cancel'):
            result = client.post(path, json=BODY, headers={**HEADERS, 'Idempotency-Key': 'same'})
            assert result.status_code == 503 and result.headers['x-should-retry'] == 'false'
            assert 'private read failure' not in result.text
        assert model.calls == []


def test_process_crash_releases_lock_but_keeps_reservation(tmp_path):
    script = '''
from pathlib import Path
import os,sys
from keyprint.replay import ReplayStore
s=ReplayStore(Path(sys.argv[1]))
s.bind({'fixture': 'crash'})
s.reserve('private-idempotency-key','digest','request-id')
os._exit(23)
'''
    result = subprocess.run([sys.executable, '-c', script, str(tmp_path)], timeout=10)
    assert result.returncode == 23
    store = ReplayStore(tmp_path)
    try:
        store.bind({'fixture': 'crash'})
        assert store.get('private-idempotency-key') == ('digest', 'request-id', None, None, None)
        assert b'private-idempotency-key' not in (tmp_path/'replay.sqlite3').read_bytes()
        assert (tmp_path/'replay.sqlite3').stat().st_mode & 0o077 == 0
    finally:
        store.close()


@pytest.mark.parametrize('name', ['replay.sqlite3', 'replay.sqlite3-journal', 'replay.lock'])
def test_symlink_store_rejected(tmp_path, name):
    target = tmp_path/'unchanged'
    target.write_text('private existing file')
    (tmp_path/name).symlink_to(target)
    with pytest.raises((OSError, ValueError)):
        ReplayStore(tmp_path)
    assert target.read_text() == 'private existing file'


def test_legacy_directory_requires_explicit_new_directory(tmp_path):
    (tmp_path/'chatcmpl-old-started.json').write_text('{}')
    with pytest.raises(ValueError, match='Legacy attempts'):
        ReplayStore(tmp_path)


def test_corrupt_response_is_not_returned_or_regenerated(tmp_path):
    headers = {**HEADERS, 'Idempotency-Key': 'corrupted'}
    with TestClient(create_app(Model, api_key=TOKEN, output=tmp_path/'runs')) as client:
        assert client.post('/v1/chat/completions', json=BODY, headers=headers).status_code == 200
    with sqlite3.connect(tmp_path/'runs/replay.sqlite3') as db:
        db.execute('UPDATE attempts SET body=?', (b'corrupted',))
    model = Model()
    with TestClient(create_app(lambda: model, api_key=TOKEN, output=tmp_path/'runs')) as client:
        result = client.post('/v1/chat/completions', json=BODY, headers=headers)
        assert result.status_code == 503 and model.calls == []
        assert 'corrupted' not in result.text
