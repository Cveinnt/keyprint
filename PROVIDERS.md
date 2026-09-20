# OpenAI and Anthropic clients

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
A different body or endpoint returns 409. Accepted work survives a disconnected or cancelled
HTTP handler, and graceful shutdown waits for it to finish. Replay works only
during this process lifetime; restarting clears that memory.
The server stops accepting new attempts after 256 records. Inspect private
artifacts before restarting; do not treat a restart as retry authorization.

## Use the Anthropic client with the same local server

```python
from pathlib import Path
from anthropic import Anthropic

client = Anthropic(
    base_url="http://127.0.0.1:8765",  # Root URL; the SDK adds /v1/messages.
    api_key=Path("local-api.key").read_bytes().hex(),
    max_retries=0,
    timeout=120,
)
message = client.messages.create(
    model="keyprint",
    max_tokens=128,
    messages=[{"role": "user", "content": "Explain why the sky is blue."}],
    extra_headers={"Idempotency-Key": "sky-messages-1"},
)
print(message.content[0].text)
print(message.stop_reason, message.usage.output_tokens)
```

This uses Keyprint's local model and watermark key. The supported
[Messages format](https://platform.claude.com/docs/en/api/messages/create)
is one user message containing a string or one text block, `model="keyprint"`,
`max_tokens` from 1 to 1024, and optional `stream=False`. The client supplies
`x-api-key` and `anthropic-version: 2023-06-01`. Raw HTTP callers must supply
those headers too. Beta features, system prompts, multiple turns, images,
tools, caching and custom sampling settings are rejected before generation.

The returned message contains local input/output token counts and the
[stop reason](https://platform.claude.com/docs/en/build-with-claude/handling-stop-reasons)
`end_turn` or `max_tokens`. These are local model counts, not Claude billing.
The two endpoints share one worker, request budget and idempotency namespace.
Use a new idempotency key when switching endpoints; an existing key cannot
return a response in the other client's format. The cancellation extension below
also accepts `x-api-key` with the same local token.

### Explicit cancellation

The local extension `POST /v1/keyprint/cancel` uses the same authentication token
and the original `Idempotency-Key` header. It is not an OpenAI-hosted API method.
From a second client or thread while the original request is in progress:

```python
import httpx

response = httpx.post(
    "http://127.0.0.1:8765/v1/keyprint/cancel",
    headers={
        "Authorization": "Bearer " + Path("local-api.key").read_bytes().hex(),
        "Idempotency-Key": "sky-example-1",
    },
    timeout=10,
)
print(response.json())
```

HTTP 202 acknowledges a cancellation request, not a stopped model. The worker
stays busy until a safe boundary before another model call or sample; an active
model call cannot be preempted. A completed result can win the race and remains
available. Otherwise the original generation ends with HTTP 410 and private
receipts retaining consumed work. Replaying that original request returns the
same terminal 410 without generating again. A new attempt requires a new key.
Cancellation after completion returns the existing terminal HTTP status without
changing the result. Unknown keys return 404. No automatic cancellation follows
from closing a browser or timing out an HTTP client.

Python callers can pass `cancel_event=threading.Event()` to `generate()` and set
the event from another thread. Keep model loading and generation on their owning
thread. Catch `KeyprintCancelled` to inspect `.report` and `.artifacts`; a
cancelled attempt is not a successful partial response. Use a fresh event for
each attempt and do not clear a requested event. This support covers the local
MLX/Transformers generation paths, not the experimental SGLang/vLLM engines.

Validated with real OpenAI 3.14.1 and Anthropic 1.6.0 clients over HTTP on pinned
Qwen/MLX experimental execution and SmolLM2/Transformers CPU. Both clients'
generated text and usage match retained private receipts; idempotency replays
the original response. Actual Anthropic-client cancellation on both backends
retains the first committed token, replays the terminal error and permits a new
32-token response on the same worker. These are bounded lifecycle checks, not
output-quality or production-load acceptance. Provider contract tests exercise
authentication, rejection, cross-protocol conflicts and shared-worker behavior.
The installed-wheel lifecycle check also forces a real OpenAI client timeout,
then verifies actual local inference, replay without duplicate generation,
busy/conflict rejection and worker reuse. Its deliberate barrier is a lifecycle
test, not a latency measurement. See [reproduction steps](INFERENCE_TESTING.md).

Transformers length completion now retains an unfinished UTF-8 suffix in
`report["carrier_rendering"]` while returning only the valid text prefix. This
can be empty at a very short cap. All committed tokens count toward usage; no
replacement character or extra model call is introduced. The report marks
full-carrier literal replay and the tokenizer rendering check unavailable for
that partial character. Complete carriers still require exact tokenizer
agreement; malformed bytes and partial characters at EOS remain errors. This
repair does not change the native SGLang/vLLM rendering paths. Default MLX
generation now separately uses reference sampling with the same retained-byte
token-limit policy. Its `keyprint.bounded-reference-report.v1` reports carry a
new runtime identity; the archived research caller remains unchanged.

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
Case, punctuation, surrounding quotes and whitespace alone do not count as a
paraphrase: an unchanged case-folded Unicode word sequence fails the lexical
screen. Changed words still do not establish preserved meaning or a watermark.

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
