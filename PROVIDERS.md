# OpenAI and Anthropic clients

Private development preview. The supported path uses provider clients with a
local model. Post-generation rewriting is blocked. Neither path inserts Keyprint
into OpenAI-hosted GPT or Anthropic-hosted Claude sampling.

The optional `clients` extra accepts `openai>=1.109.1,<4` and
`anthropic>=0.83.0,<2`, so installing Keyprint need not replace an existing
compatible client. The server itself does not need either client package;
omit `clients` if your application already manages those dependencies.
Numerical and model-backend dependencies retain their separate pinned contracts.

Two client pairs have been tested locally: OpenAI 1.109.1 with Anthropic 0.83.0,
and OpenAI 3.14.1 with Anthropic 1.6.0. These are tested endpoints within the
allowed ranges, not a test of every intervening or future release. The older
pair passes actual Qwen text cancellation/replay and typed JSON parsing/replay.
Its structured run reconciles eleven requests, 213 tokens and 175 grammar masks.
Anthropic 0.83.0 emits Pydantic serialization warnings for parsed response objects;
the parsed values and replay checks pass, and the warnings remain in the log.
See [local version-matrix instructions](CONTRIBUTING.md#client-version-checks).

See [all runtime/client combinations](COMPATIBILITY.md) and the tested
[LangChain example](examples/langchain_local.py).

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
HTTP handler, and graceful shutdown waits for it to finish.

The new durable-replay implementation is **pending local test validation**.
It records an accepted request before generation and stores the exact terminal
response in a private SQLite journal under `--output`. Restarting with the same
directory, model identity, watermark key, API token and server implementation
replays completed successes or failures without generating again. An attempt
without a durable terminal response returns 410; it is never silently restarted.
Use an explicit `Idempotency-Key` before sending a request: a disconnected client
cannot recover a server-generated ID it never received.

The request limit is 256 accepted attempts per directory, including failures;
restarting does not reset it. A different request using an existing key returns
409. One process owns each directory. Model/key/server changes fail startup;
use a separate directory for new work and preserve the old one. Legacy directories
without the replay journal are rejected instead of pretending old requests can
be recovered. Never reissue an uncertain attempt with a new key to bypass recovery.

This implementation requires a local POSIX filesystem and protects the journal
with mode 600 inside the mode-700 output directory. It stores response text:
keep it private with the generation artifacts. Network filesystems, multi-host
failover, power-loss durability and Windows serving are not qualified. Disk or
integrity failures refuse work; they do not authorize a retry. No hosted-provider
watermarking or production qualification is implied.

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

## Typed JSON output

Install `.[server,clients,transformers,structured]` and start the Transformers
server above with `--model models/smollm3`. The tested OpenAI and Anthropic
clients can parse a shared Pydantic type directly:

The same client code also works with the pinned MLX server. Install
`.[server,clients,mlx,structured]` and use `--backend mlx` with the pinned Qwen
model directory. The grammar and validation contract are shared; model output
quality and watermark capacity must be evaluated separately.

```python
from pathlib import Path
from pydantic import BaseModel, ConfigDict
from openai import OpenAI
from anthropic import Anthropic

class Record(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    count: int
    enabled: bool

token = Path("local-api.key").read_bytes().hex()
messages = [{"role": "user", "content": "Return JSON: name Maya, count 3, enabled false."}]

with OpenAI(base_url="http://127.0.0.1:8765/v1", api_key=token,
            max_retries=0, timeout=180) as client:
    response = client.chat.completions.parse(
        model="keyprint", messages=messages, max_completion_tokens=128,
        response_format=Record, extra_headers={"Idempotency-Key": "record-openai-1"},
    )
    print(response.choices[0].message.parsed)

with Anthropic(base_url="http://127.0.0.1:8765", api_key=token,
               max_retries=0, timeout=180) as client:
    response = client.messages.parse(
        model="keyprint", messages=messages, max_tokens=128,
        output_format=Record, extra_headers={"Idempotency-Key": "record-anthropic-1"},
    )
    print(response.parsed_output)
```

OpenAI uses `response_format={"type":"json_schema","json_schema":
{"name":"record","strict":true,"schema":...}}`; Anthropic uses
`output_config={"format":{"type":"json_schema","schema":...}}`. These are
local protocol implementations. They do not watermark hosted GPT or Claude.
The server passes the schema to the model adapter and binds it into the private
generation receipt. Idempotency covers the schema as part of the request body.

Supported schemas have a top-level object and use types, properties, required
fields, additionalProperties, items, array/string length bounds, numeric bounds,
enum/const and supported anyOf/allOf/oneOf combinations. Compilation warnings
or unsupported combinations fail instead of silently weakening the constraint.
References (including `$defs`), regex, formats, custom extensions and dialect
overrides are currently rejected. Schemas are limited to 32 KiB and 24 nested
schema levels; the parser also has explicit resource limits. This is a bounded
subset, not complete JSON Schema support.

The grammar mask runs before top-k filtering and watermark sampling. Every
token still has a model forward, a sample and a durable commit; there is no
forced-token fast-forward, rollback, postprocessing or hidden retry. Complete
outputs undergo independent JSON/schema validation, including duplicate-key
rejection. Truncation returns `length`/`max_tokens` with
`keyprint.structured_output.schema_validated=false`; typed client helpers may
raise instead of returning a parsed object. Treat this as incomplete, not as a
validated record. The SDK's report carries the same status.

Pinned SmolLM3 local testing completed six constrained ordinary/marked outputs,
two typed-client requests and one deliberate token cap, alongside two
unconstrained controls. Both controls retained prohibited Markdown fences.
The constrained JSON and Unicode cases matched their requested values, and
both negation extractions preserved the prerequisite in this small sample.
These checks establish the tested format/client path, not semantic reliability
or detection power. Constraints can leave little or no marking capacity.
Pinned Qwen/MLX subsequently passed the same eleven-request suite on each of
the default reference and experimental-fast paths. Across both runs, 426
generated tokens and 350 grammar masks reconcile with private journals;
both typed clients parse and replay exactly. All six constrained examples
per path reach EOS, while the deliberate one-token cap remains incomplete.
The sampled multilingual values and backup prerequisite match the prompts.
These are local client calls to Qwen, not hosted GPT or Claude generation.
Native SGLang/vLLM, tools and streaming do not yet support this mode.

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

## PydanticAI

Start the local server above, then install the application client separately:

```sh
python -m pip install 'pydantic-ai-slim[openai]==2.51.0'
python examples/pydantic_ai_local.py
```

The [recipe](examples/pydantic_ai_local.py) uses `OpenAIChatModel` with your local
`AsyncOpenAI` client, disables automatic retries and sets an idempotency key for
each logical request. `agent.run(...)` and `agent.run_sync(...)` are tested.
In PydanticAI 2.51.0, token accounting is `result.usage.input_tokens` and
`result.usage.output_tokens`; `usage` is a property.

For a typed response, pass a Pydantic model to the recipe's
`local_agent(client, output_type=YourModel)`. It explicitly selects
`NativeOutput(..., strict=True, template=False)`. This requires the server's
`structured` extra and a supported structured-output backend. Actual PydanticAI
testing covers the pinned Qwen/MLX setup. Ordinary `Agent(..., output_type=YourModel)`
selects tool output under this profile and is rejected; tool calls are not
implemented by this endpoint.

Keep each request to one user message. Instructions/system prompts, message
history and streaming reject before inference. Reuse a request ID only with the
identical request; conflicting reuse returns 409. This recipe does not watermark
hosted OpenAI responses, add tools or enable autonomous agent loops.

[Actual ordinary/marked samples, replay checks and install audit](evidence/pydantic-ai-2026-09-28/README.md).
The recipe follows PydanticAI's [custom OpenAI client](https://pydantic.dev/docs/ai/models/openai/)
and [native output](https://pydantic.dev/docs/ai/core-concepts/output/) interfaces.

## Existing GPT or Claude text

`rewrite()`, `rewrite_openai()` and `rewrite_anthropic()` are unavailable. They
raise the public `RewriteUnavailableError` before inference or output creation.
Provider responses are not modified. There is no retry, automatic translation,
unchecked candidate return or fallback claiming the unchanged original is marked.

The earlier rewriting pipeline produced unrequested translations and changed
approval conditions. Literal checks and same-language prompts cannot guarantee
preservation. An experimental label does not make those outputs acceptable.
Historical samples remain in [QUALITY_REVIEW.md](QUALITY_REVIEW.md); they are
failure evidence, not examples of a supported feature. Historical rewrite tools
now encounter this same rejection rather than bypassing it.

Keep existing documents unchanged. To create a new response, use the local
client examples above, or `generate()`. This is a distinct operation and does not
solve post-generation watermarking. Native output quality remains unqualified;
this safety restriction does not close the quality requirement or enable launch.


## Ollama Python client, local Keyprint worker

The optional client requires a separate `pip install ollama==0.6.2`. Keyprint
adds no Ollama or LiteLLM dependency to its core or server extras. Use the same
`keyprint serve` command and local API key described above.

[Runnable example](examples/ollama_local.py). `Client.chat()` and
`AsyncClient.chat()` accept one user text message; `Client.generate()` accepts
a prompt. Set `model="keyprint"`, `stream=False`, and
`options={"num_predict": 96}` (1–1,024). The official client sends empty tools
by default; only that empty list is accepted. Nonempty tools, streaming, images,
model downloads, `keep_alive`, thinking, raw/template modes, custom sampling,
JSON formats and multi-turn conversations are not supported. Unknown options
fail before inference. The shared cancellation endpoint and process-local
idempotency rules apply across all three client protocols.

This is **Ollama client compatibility with Keyprint's local worker**, not a
plugin inside the Ollama daemon. It neither installs Ollama nor claims that
pointing Keyprint at an arbitrary Ollama/LM Studio endpoint inserts a sampler.
The tested native worker was Llama 3.2 3B Q8_0 through llama.cpp on ARM64 CPU.
Three ordinary/marked pairs, sync generate/chat, async chat, exact replay and
text/usage reconciliation passed. All outputs are retained in
[evidence](evidence/ollama-client-2026-09-21/results.json), including added detail.
Quality and detection were not accepted. Other backend/client combinations
remain untested even when the server routes share code.

## Routing without duplicating or losing watermarking

[LiteLLM example](examples/litellm_local.py), tested with 1.102.0. Two explicit
local routes passed sync/async requests, forwarded idempotency headers, isolated
workers and replayed exact text/usage. A deliberate pre-inference barrier caused
an actual client timeout; the accepted attempt was recovered without a second
generation or fallback. Unsupported tools, running-attempt conflicts and busy
responses did not dispatch to the other worker. Five actual outputs and the
failed first harness attempt remain in [evidence](evidence/router-2026-09-21/).

Keep routing outside Keyprint. Start with one endpoint per alias, no automatic
retries, no fallback, and no response cache. Preserve the same worker, request
body and idempotency key for recovery. A different worker has no shared replay
record, even if it uses the same model/key. A timeout is not proof no tokens
were generated. The working configuration is an explicit-route pilot, not
qualification of LiteLLM Proxy, auto-routing, failover or distributed replay.

Future automatic routing must select only approved marked backends, retain the
selected identity with the response, and keep accepted work on its original
worker. It must reject unavailable routes rather than silently return hosted or
unmarked output as watermarked. No external tracing or prompt logging was enabled
in these tests. Do not globally enable parameter dropping to hide incompatibility.
