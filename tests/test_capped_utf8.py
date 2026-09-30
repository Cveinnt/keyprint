import json

import pytest

from keyprint import KeyprintError
from keyprint.experimental.fast_mlx import pipeline
from keyprint._engine.legacy._impl.research.token_channel_host import EOS, THINK_OPEN, TOOL_OPEN
from test_fast_caller import Backend, candidate, head as model_head
from test_fast_mlx import head


@pytest.mark.parametrize('condition', ['ordinary', 'marked'])
@pytest.mark.parametrize('token,pending', [
    (126, 'c2'), (158, 'e2'), (37698, 'e29c'),
    (172, 'f0'), (9284, 'f09f'), (92173, 'f09f92'), (25521, 'e29c'),
])
def test_length_cap_retains_every_byte_and_reports_partial_text(tmp_path, condition, token, pending):
    kp = candidate()
    calls = []
    def model(ids, **_):
        calls.append(1)
        return model_head(ids, (32 if len(calls) <= 2 else token,))
    result = kp._run(model, [32, 32], max_tokens=2, condition=condition,
                     output=tmp_path / 'capped', backend=Backend, cache_factory=lambda _: [])
    report = result.report
    assert result.text == ('A ' if token == 25521 else 'A')
    assert len(calls) == 3  # One prefill, exactly two generation forwards.
    assert report['payload']['completion'] == 'length'
    assert report['payload']['committed_token_ids'] == [32, token]
    assert report['usage']['completion_tokens'] == 2
    assert report['carrier_rendering'] == [{
        'channel': 'visible', 'status': 'incomplete_utf8_at_token_limit',
        'pending_utf8_hex': pending, 'committed_token_ids': (32, token),
    }]
    assert report['literal_replay_status'][0]['availability'] == 'unavailable'
    assert report['payload']['literal_diagnostics'][0]['events'] is None
    saved = json.loads((result.artifacts / 'report.json').read_text())['report']
    assert saved['carrier_rendering'][0]['pending_utf8_hex'] == pending
    followup = kp._run(lambda ids, **_: model_head(ids, (32,)), [32], max_tokens=1,
                       condition=condition, output=tmp_path / 'next', backend=Backend,
                       cache_factory=lambda _: [])
    assert followup.text == 'A'
    assert followup.report['carrier_rendering'][0]['status'] == 'complete_utf8'


@pytest.mark.parametrize('ending', [None, EOS, THINK_OPEN, TOOL_OPEN])
def test_non_length_boundaries_still_reject_pending_utf8(ending):
    with pipeline(bytes(range(32)), allow_thinking=True, allow_tools=True) as p:
        p.step(head([25521]), lambda _: 0)
        with pytest.raises(UnicodeDecodeError):
            if ending is None:
                p.finish()
            else:
                p.step(head([ending]), lambda _: 0)
        assert p.receipt()['final'] is None
        assert p.committed_token_ids == ((25521,) if ending is None else (25521, ending))


@pytest.mark.parametrize('token', [187, 102])
def test_invalid_utf8_is_not_reclassified_as_length(tmp_path, token):
    with pytest.raises(KeyprintError) as caught:
        candidate()._run(lambda ids, **_: model_head(ids, (token,)), [32], max_tokens=1,
                         condition='ordinary', output=tmp_path / 'invalid', backend=Backend,
                         cache_factory=lambda _: [])
    assert caught.value.report['payload']['committed_tokens'] == 1


def test_complete_multibyte_character_and_exact_limit_contract():
    with pipeline(bytes(range(32))) as p:
        p.step(head([37698]), lambda _: 0)
        for invalid in (True, 0, 2):
            with pytest.raises(ValueError, match='exact reached token cap'):
                p.finish_at_limit(invalid)
        p.step(head([241]), lambda _: 0)
        result = p.finish_at_limit(2)
        assert result.visible.text == '\u2713'
        assert not hasattr(result.visible, 'pending_utf8_hex')
        assert p.receipt()['finalized'] is True


@pytest.mark.parametrize('channel', [THINK_OPEN, TOOL_OPEN])
def test_unclosed_generated_channel_retains_pending_bytes(channel):
    with pipeline(bytes(range(32)), allow_thinking=True, allow_tools=True) as p:
        p.step(head([channel]), lambda _: 0)
        p.step(head([25521]), lambda _: 0)
        result = p.finish_at_limit(2)
        carrier = result.reasoning if channel == THINK_OPEN else result.tools[0]
        assert result.protocol_complete is False
        assert result.visible.text == ''
        assert carrier.text == ' ' and carrier.pending_utf8_hex == 'e29c'
        assert result.committed_token_ids == (channel, 25521)


def test_zero_token_limit_does_not_load_cache_or_call_model(tmp_path):
    def forbidden(*args, **kwargs):
        raise AssertionError('Zero-token response must not run inference')
    result = candidate()._run(forbidden, [32], max_tokens=0, condition='marked',
                              output=tmp_path / 'zero', backend=Backend, cache_factory=forbidden)
    assert result.text == ''
    assert result.report['payload']['completion'] == 'length'
    assert result.report['payload']['committed_token_ids'] == []


def test_openai_client_receives_length_and_replays_without_regeneration(tmp_path):
    pytest.importorskip('fastapi')
    pytest.importorskip('openai')
    from fastapi.testclient import TestClient
    from openai import OpenAI
    from keyprint.server import create_app
    calls = []
    class LocalModel:
        identity = {"fixture": "capped-utf8"}
        _key = bytes(32)
        def __init__(self):
            self.sdk = candidate()

        def generate(self, prompt, *, max_tokens, output, cancel_event=None):
            calls.append(prompt)
            return self.sdk._run(lambda ids, **_: model_head(ids, (25521,)), [32],
                max_tokens=max_tokens, condition='marked', output=output, backend=Backend,
                cache_factory=lambda _: [], cancel_event=cancel_event)
    token = 'utf8-fixture-auth-token-' * 3
    with TestClient(create_app(LocalModel, api_key=token, output=tmp_path / 'http')) as transport:
        with OpenAI(api_key=token, base_url='http://testserver/v1', http_client=transport,
                    max_retries=0) as client:
            settings = dict(model='keyprint', messages=[{'role': 'user', 'content': 'UTF-8 fixture'}],
                            max_tokens=1, extra_headers={'Idempotency-Key': 'capped-utf8-test'})
            first = client.chat.completions.create(**settings)
            second = client.chat.completions.create(**settings)
            assert first.model_dump() == second.model_dump()
            assert first.choices[0].message.content == ' '
            assert first.choices[0].finish_reason == 'length'
            assert first.usage.completion_tokens == 1
    assert calls == ['UTF-8 fixture']
