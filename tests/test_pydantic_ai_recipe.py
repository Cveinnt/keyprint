import asyncio
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path

import pytest

pytest.importorskip('pydantic_ai')
import httpx
from openai import AsyncOpenAI
from pydantic import BaseModel, ConfigDict
from keyprint.server import CompletionRequest

spec = importlib.util.spec_from_file_location('pydantic_ai_recipe',
    Path(__file__).parents[1] / 'examples/pydantic_ai_local.py')
recipe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recipe)


class Approval(BaseModel):
    model_config = ConfigDict(extra='forbid')
    name: str
    approved: bool


@pytest.mark.parametrize('structured', [False, True])
def test_official_client_request_fits_strict_server_contract(structured):
    captured = []
    def reply(request):
        body = json.loads(request.content)
        CompletionRequest.model_validate(body)
        assert request.url.path == '/v1/chat/completions'
        assert request.headers['Idempotency-Key'] == 'fixed-attempt'
        captured.append(body)
        content = '{"name":"Maya","approved":false}' if structured else 'A seed needs water to germinate.'
        return httpx.Response(200, json={'id': 'chatcmpl-local', 'object': 'chat.completion', 'created': 1,
            'model': 'keyprint', 'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': content}, 'finish_reason': 'stop'}],
            'usage': {'prompt_tokens': 10, 'completion_tokens': 9, 'total_tokens': 19}})
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(reply)) as http:
            async with AsyncOpenAI(base_url='http://127.0.0.1/v1', api_key='local-test', http_client=http, max_retries=0) as client:
                agent = recipe.local_agent(client, output_type=Approval if structured else str)
                result = await agent.run('Return the requested output.', model_settings={
                    'max_tokens': 96, 'extra_headers': {'Idempotency-Key': 'fixed-attempt'}})
                assert asdict(result.usage)['input_tokens'] == 10
                assert result.usage.output_tokens == 9
                assert result.usage.requests == 1
                if structured:
                    assert result.output == Approval(name='Maya', approved=False)
                else:
                    assert result.output == 'A seed needs water to germinate.'
    asyncio.run(run())
    assert len(captured) == 1
    assert ('response_format' in captured[0]) is structured
    assert 'tools' not in captured[0]


def test_provider_errors_do_not_start_hidden_retries():
    from pydantic_ai.exceptions import ModelHTTPError
    calls = []
    def fail(request):
        calls.append(request)
        return httpx.Response(503, headers={'x-should-retry': 'false'}, json={'error': {'message': 'busy', 'type': 'server_error'}})
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(fail)) as http:
            async with AsyncOpenAI(base_url='http://127.0.0.1/v1', api_key='local-test', http_client=http, max_retries=0) as client:
                with pytest.raises(ModelHTTPError) as caught:
                    await recipe.local_agent(client).run('Hello', model_settings={'max_tokens': 10})
                assert caught.value.status_code == 503
    asyncio.run(run())
    assert len(calls) == 1
