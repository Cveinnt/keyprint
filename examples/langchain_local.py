"""Text-only LangChain request to an already running local Keyprint server.

Install langchain-openai==1.6.2 separately. Start the server as documented in
PROVIDERS.md. local-api.key is the API authentication key, not the watermark key.
"""
import os
from pathlib import Path
from uuid import uuid4

os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"

from langchain_openai import ChatOpenAI

chat = ChatOpenAI(
    model="keyprint",
    base_url="http://127.0.0.1:8765/v1",
    api_key=Path("local-api.key").read_bytes().hex(),
    use_responses_api=False,
    temperature=None,  # Use the server's fixed sampling configuration.
    max_tokens=96,
    max_retries=0,
    timeout=180,
)
request_id = str(uuid4())
response = chat.invoke(
    "Explain in one sentence why a seed needs water.",
    extra_headers={"Idempotency-Key": request_id},
)
print(response.content)
# Reuse request_id and the same prompt only to recover this same attempt.
# await chat.ainvoke(...) supports the same text-only request and headers.
