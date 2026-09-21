"""Ollama Python client calling Keyprint, not the Ollama daemon.

Install ollama==0.6.2 separately and start Keyprint as in PROVIDERS.md.
"""
from pathlib import Path
from uuid import uuid4
from ollama import Client

request_id = str(uuid4())
client = Client(
    host="http://127.0.0.1:8765",
    headers={
        "Authorization": "Bearer " + Path("local-api.key").read_bytes().hex(),
        "Idempotency-Key": request_id,
    },
    timeout=180,
)
try:
    response = client.chat(
        model="keyprint",
        messages=[{"role": "user", "content": "Explain why a seed needs water."}],
        options={"num_predict": 96},
        stream=False,
    )
    print(response.message.content)
finally:
    client.close()
# For another experiment, use a new request_id. Reuse the same ID and body
# only to recover this attempt in the same server process.
