"""PydanticAI against a running local Keyprint server, with explicit replay IDs.

Install pydantic-ai-slim[openai]==2.51.0 separately. Start the local server as
described in PROVIDERS.md. NativeOutput additionally requires a Keyprint server
with the structured extra and its supported MLX or Transformers backend.
"""
import asyncio
from pathlib import Path
from uuid import uuid4

from openai import AsyncOpenAI
from pydantic_ai import Agent, NativeOutput
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.profiles.openai import OpenAIModelProfile
from pydantic_ai.providers.openai import OpenAIProvider


def local_agent(client: AsyncOpenAI, *, output_type=str) -> Agent:
    """One-turn text or explicit native JSON, without tools or automatic retries."""
    model = OpenAIChatModel(
        'keyprint', provider=OpenAIProvider(openai_client=client),
        profile=OpenAIModelProfile(supports_tools=False, supports_json_schema_output=True),
    )
    output = str if output_type is str else NativeOutput(output_type, strict=True, template=False)
    return Agent(model, output_type=output, retries=0)


async def main():
    # This authenticates to your local server; it is not your watermark key.
    async with AsyncOpenAI(
        base_url='http://127.0.0.1:8765/v1',
        api_key=Path('local-api.key').read_bytes().hex(),
        max_retries=0, timeout=180,
    ) as client:
        agent = local_agent(client)
        settings = {'max_tokens': 96, 'extra_headers': {'Idempotency-Key': str(uuid4())}}
        result = await agent.run('Explain in one sentence why a seed needs water.', model_settings=settings)
        print(result.output)
        # Reuse the same ID and identical request only to recover this attempt.
        # No system prompts, conversation history, tool calls or streaming yet.


if __name__ == '__main__':
    asyncio.run(main())
