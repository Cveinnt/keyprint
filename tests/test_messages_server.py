"""Anthropic client contract and shared-worker lifecycle; fixture generation."""
from concurrent.futures import ThreadPoolExecutor

import pytest

pytest.importorskip('fastapi')
anthropic = pytest.importorskip('anthropic')
from fastapi.testclient import TestClient
from keyprint import Generation
from keyprint.server import create_app
from test_server import Model, TOKEN, HEADERS, BODY

MESSAGES = {**BODY, 'max_tokens': 8}
AUTH = {'x-api-key': TOKEN, 'anthropic-version': '2023-06-01'}


@pytest.mark.parametrize('completion,stop', [('eos', 'end_turn'), ('length', 'max_tokens')])
@pytest.mark.parametrize('content', ['hello', [{'type': 'text', 'text': 'hello'}]])
def test_real_client_schema_usage_request_id_and_replay(tmp_path, completion, stop, content):
    class ResultModel(Model):
        def generate(self, *args, **kwargs):
            result = super().generate(*args, **kwargs)
            return Generation(result.text, {**result.report, 'completion': completion}, result.artifacts)
    model = ResultModel()
    with TestClient(create_app(lambda: model, api_key=TOKEN, output=tmp_path / 'http')) as transport:
        with anthropic.Anthropic(api_key=TOKEN, base_url='http://testserver', http_client=transport,
                                 max_retries=0, _strict_response_validation=True) as client:
            params = {**MESSAGES, 'messages': [{'role': 'user', 'content': content}],
                      'extra_headers': {'Idempotency-Key': 'messages-one'}}
            first = client.messages.create(**params)
            assert first.type == 'message' and first.role == 'assistant'
            assert first.model == 'keyprint' and first.content[0].text == 'hello'
            assert first.stop_reason == stop and first.stop_sequence is None
            assert first.usage.input_tokens == 4 and first.usage.output_tokens == 1
            assert first._request_id == first.id and first.id.startswith('msg_')
            assert first.model_extra['keyprint']['hosted_provider'] is False
            assert client.messages.create(**params).model_dump() == first.model_dump()
    assert model.calls == ['hello']


@pytest.mark.parametrize('change', [
    {'stream': True}, {'stream': 0}, {'max_tokens': True}, {'max_tokens': 0}, {'max_tokens': 1025},
    {'model': 'claude-anything'}, {'tools': []}, {'system': 'hidden system'}, {'temperature': .5},
    {'thinking': {'type': 'enabled', 'budget_tokens': 4}}, {'stop_sequences': ['STOP']},
    {'messages': [{'role': 'assistant', 'content': 'prefix'}]},
    {'messages': [BODY['messages'][0], BODY['messages'][0]]},
    {'messages': [{'role': 'user', 'content': []}]},
    {'messages': [{'role': 'user', 'content': [{'type': 'image', 'source': {}}]}]},
    {'messages': [{'role': 'user', 'content': [{'type': 'text', 'text': 'hello', 'cache_control': {'type': 'ephemeral'}}]}]},
    {'messages': [{'role': 'user', 'content': [{'type': 'text', 'text': 'a'}, {'type': 'text', 'text': 'b'}]}]},
])
def test_unsupported_messages_fail_before_model(tmp_path, change):
    model = Model()
    with TestClient(create_app(lambda: model, api_key=TOKEN, output=tmp_path / 'http')) as client:
        response = client.post('/v1/messages', json={**MESSAGES, **change}, headers=AUTH)
        assert response.status_code == 400
        assert response.json()['type'] == 'error'
        assert response.json()['error']['type'] == 'invalid_request_error'
        assert response.headers['x-should-retry'] == 'false'
    assert model.calls == []


def test_auth_version_body_limits_and_conflicting_credentials(tmp_path):
    model = Model()
    with TestClient(create_app(lambda: model, api_key=TOKEN, output=tmp_path / 'http')) as client:
        for headers in ({}, {**AUTH, 'x-api-key': 'wrong'}, {**AUTH, 'Authorization': 'Bearer wrong'}):
            response = client.post('/v1/messages', json=MESSAGES, headers=headers)
            assert response.status_code == 401
            assert response.json()['error']['type'] == 'authentication_error'
        for headers in ({'x-api-key': TOKEN}, {**AUTH, 'anthropic-version': 'new'}, {**AUTH, 'anthropic-beta': 'unsupported'}):
            assert client.post('/v1/messages', json=MESSAGES, headers=headers).status_code == 400
        assert client.post('/v1/messages', json=BODY, headers=AUTH).status_code == 400
        response = client.post('/v1/messages', content=b' ' * 131073,
                               headers={**AUTH, 'Content-Type': 'application/json'})
        assert response.status_code == 413 and response.json()['error']['type'] == 'request_too_large'
        assert client.post('/v1/messages', content='text', headers=AUTH).status_code == 415
        assert model.calls == []
        assert client.post('/v1/messages', json=MESSAGES, headers={**HEADERS, 'anthropic-version': '2023-06-01'}).status_code == 200


def test_cross_protocol_idempotency_and_request_budget(tmp_path):
    model = Model()
    with TestClient(create_app(lambda: model, api_key=TOKEN, output=tmp_path / 'http', max_requests=1)) as client:
        headers = {**AUTH, 'Idempotency-Key': 'shared-key'}
        first = client.post('/v1/messages', json=MESSAGES, headers=headers)
        assert first.status_code == 200
        assert client.post('/v1/chat/completions', json=MESSAGES, headers={**HEADERS, 'Idempotency-Key': 'shared-key'}).status_code == 409
        conflict = client.post('/v1/messages', json={**MESSAGES, 'max_tokens': 9}, headers=headers)
        assert conflict.status_code == 409 and conflict.json()['type'] == 'error'
        assert client.post('/v1/messages', json=MESSAGES, headers=headers).json() == first.json()
        assert client.post('/v1/messages', json=MESSAGES, headers=AUTH).status_code == 429
        assert model.calls == ['hello']


def test_cross_protocol_cancellation_and_worker_reuse(tmp_path):
    model = Model(block=True)
    headers = {**AUTH, 'Idempotency-Key': 'cancel-messages'}
    with TestClient(create_app(lambda: model, api_key=TOKEN, output=tmp_path / 'http')) as transport:
        with anthropic.Anthropic(api_key=TOKEN, base_url='http://testserver', http_client=transport,
                                 max_retries=0) as client:
            def request():
                with pytest.raises(anthropic.APIStatusError) as caught:
                    client.messages.create(**MESSAGES, extra_headers={'Idempotency-Key': 'cancel-messages'})
                return caught.value
            with ThreadPoolExecutor(max_workers=1) as worker:
                first = worker.submit(request)
                try:
                    assert model.started.wait(5)
                    assert transport.post('/v1/messages', json=MESSAGES, headers=headers).status_code == 409
                    assert transport.post('/v1/chat/completions', json=BODY, headers=HEADERS).status_code == 503
                    assert transport.post('/v1/keyprint/cancel', headers=headers).status_code == 202
                    assert transport.post('/v1/messages', json=MESSAGES, headers=AUTH).status_code == 503
                finally:
                    model.release.set()
                stopped = first.result()
                assert stopped.status_code == 410 and stopped.response.json()['type'] == 'error'
            assert request().response.json() == stopped.response.json()
            assert model.calls == ['hello']
            assert transport.post('/v1/chat/completions', json=BODY, headers=HEADERS).status_code == 200
            assert model.calls == ['hello', 'hello']


def test_client_error_replays_without_retry_or_private_exception(tmp_path):
    model = Model(fail=True)
    with TestClient(create_app(lambda: model, api_key=TOKEN, output=tmp_path / 'http')) as transport:
        with anthropic.Anthropic(api_key=TOKEN, base_url='http://testserver', http_client=transport) as client:
            errors = []
            for _ in range(2):
                with pytest.raises(anthropic.InternalServerError) as caught:
                    client.messages.create(**MESSAGES, extra_headers={'Idempotency-Key': 'failed'})
                errors.append(caught.value.response.json())
                assert 'private failure' not in caught.value.response.text
                assert caught.value.request_id == errors[-1]['request_id']
            assert errors[0] == errors[1]
    assert model.calls == ['hello']  # Server forbids retries even with SDK defaults.
