# OpenAI client and completed provider responses

Private development preview. These are two different integration paths. Neither
inserts Keyprint into OpenAI-hosted GPT or Anthropic-hosted Claude sampling.

## Use the OpenAI client with a local model

From this checkout, install the server and one backend. Download the pinned
model using the [README](README.md) instructions first.

```sh
pip install '.[server,clients,transformers]'
keyprint keygen keyprint.key
keyprint keygen local-api.key
keyprint serve --backend transformers --model models/smollm2 \
  --key keyprint.key --api-key local-api.key --output private-keyprint-server
```

On Apple Silicon use the `mlx` extra, `--backend mlx`, and the exact Qwen model
path instead. The server binds only to `127.0.0.1:8765` and loads the model once.
In a second terminal, with the same environment and working directory:

```python
from pathlib import Path
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8765/v1",
    api_key=Path("local-api.key").read_bytes().hex(),
    max_retries=0,
    timeout=120,
)
result = client.chat.completions.create(
    model="keyprint",
    messages=[{"role": "user", "content": "Explain why the sky is blue."}],
    max_completion_tokens=128,
    extra_headers={"Idempotency-Key": "sky-example-1"},
)
print(result.choices[0].message.content)
```

The API authentication token is separate from the watermark key. Never send the
watermark key in a client request. Only one user text message is supported.
Streaming, tools, system messages, multiple turns, logprobs, custom sampling
parameters and multiple choices are rejected before generation. This is a local
Chat Completions subset, not a full OpenAI API replacement or production server.

Requests run on one model worker. Busy requests return 503 without starting
generation. Use no automatic retries: a timeout does not cancel the model run.
Repeat the same idempotency key and body to recover the accepted attempt: 409
means it is still running; after completion, the original response is replayed.
A different body returns 409. Accepted work survives a disconnected or cancelled
HTTP handler, and graceful shutdown waits for it to finish. This is result
recovery, not model cancellation. Replay works only during this process lifetime;
restarting clears that memory.
The server stops accepting new attempts after 256 records. Inspect private
artifacts before restarting; do not treat a restart as retry authorization.

Validated with a real OpenAI 3.14.1 client, HTTP socket, pinned Qwen/MLX generation
and identical-response idempotency replay. Provider SDK contract tests exercise
authentication, rejection, single-worker ownership and failure behavior.
The installed-wheel lifecycle check also forces a real OpenAI client timeout,
then verifies actual local inference, replay without duplicate generation,
busy/conflict rejection and worker reuse. Its deliberate barrier is a lifecycle
test, not a latency measurement. See [reproduction steps](INFERENCE_TESTING.md).

## Inspect a local rewrite of GPT or Claude prose

For an **already completed** provider result, load a local model and explicitly
request a second generation. No provider request is made by these helpers:

```python
from pathlib import Path
from keyprint import Keyprint

watermark = Keyprint.from_mlx(
    "models/qwen3-8b-4bit", key=Path("keyprint.key").read_bytes(),
)
# `response` is an existing OpenAI ChatCompletion or completed Responses result.
comparison = watermark.rewrite_openai(response, max_tokens=384)
# For an existing Anthropic Message: watermark.rewrite_anthropic(message)
print(comparison.original)
print(comparison.text)
print(comparison.status, comparison.checks)
```

Plain prose can also use `watermark.rewrite(text)`. The returned text is always
a candidate. `failed_checks` means a lexical or completion check failed;
`needs_review` means those checks passed, **not** that meaning, quality or a
detectable watermark was verified. There is no automatic approved status.
Both texts and checks are saved in the private generation directory. Failed or
unchanged outputs are retained without retry or substitution.

The helpers reject provider responses containing tool use, thinking, citations,
refusals, incomplete outputs or supported structured-output metadata. Obvious
JSON/code-fenced prose is rejected too; that heuristic cannot classify all code.
Do not send structured outputs, code or consequential content through this
experimental path. Numbers, URLs and email addresses receive exact lexical
checks; negation, names, translation and factual meaning require separate review.

Tests construct actual OpenAI and Anthropic SDK response objects without hosted
API calls. A paired local screen covers English email, negation and technical
prose, plus Spanish, French and Chinese. It found unchanged candidates, escaped
line breaks, numeric-format changes and wording changes needing review. The
second prompt produced changed wording in all 12 runs; two French runs changed
`09:30` to `09h30`. That is a strict fidelity failure, not evidence of a new time.
These results do not establish negligible quality loss or watermark detection.

A third prompt explicitly requested real line breaks. All 12 candidates changed
wording and passed the escaped-line-break check; both French outputs still
failed exact numeric preservation. All three screens are retained, including
failed outputs. They are prompt-development runs, not a held-out quality study.

The mode adds local model latency and may alter meaning. It has no transferred
acceptances from the reference research ledger. Native hosted-provider sampling
integration and a broadly usable, validated hosted-output product remain open.
