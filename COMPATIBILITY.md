# Compatibility at a glance

Updated September 28, 2026. Applies to the **private, unpublished Keyprint
0.1.0a1 development branch**. “Tested” means actual inference through the
listed local path. It does not mean production qualification, arbitrary model
support or transfer of the original research acceptances.

| Runtime | Keyprint path | Tested model / environment | OpenAI Python | Anthropic Python | LangChain | Limits |
| --- | --- | --- | --- | --- | --- | --- |
| MLX | Tested local | Qwen3-8B · Apple Silicon | Tested | Tested | Not tested | Single response; typed JSON tested. Production batching unqualified. |
| Transformers | Tested local | SmolLM2 / SmolLM3 · CPU | Tested | Tested | Not tested | Single response; typed JSON tested. Other tokenizer families unqualified. |
| llama.cpp / GGUF | Tested local | SmolLM2 / Llama 3.2 3B · CPU | Tested | Tested | Tested on Llama | LangChain sync + async; no tools, streaming or constrained JSON. |
| SGLang | Experimental hook | Pinned modified ARM CPU build · SmolLM2 | Not tested | Not tested | Not tested | 620 returned tokens matched journals. Server lifecycle and GPU unqualified. |
| vLLM | Experimental native server | 0.29.0+cpu · SmolLM2 | Tested native HTTP | Tested native HTTP | Not tested | Concurrent requests, disconnect and recovery tested; GPU and production remain unqualified. |
| Ollama | No Keyprint adapter | Upstream OpenAI-compatible API | Not integrated | Not integrated | Not integrated | An API-compatible endpoint does not expose the required native sampling hook. |
| Hosted GPT / Claude | No Keyprint sampling hook | Provider-hosted generation | Client only | Client only | Not integrated | Local client compatibility does not watermark hosted GPT or Claude. Rewriting is blocked. |

Both provider clients connect to **Keyprint's local model**. They do not add
Keyprint to hosted GPT/Claude generation. Supported local requests contain one
user text message, a token cap and an idempotency key. Tools, system/multi-turn
messages and streaming are rejected before inference. Replay lasts only for
the server process. See [provider usage](PROVIDERS.md) for exact boundaries.

## Newer Qwen model boundary

Qwen3.5-9B now has a separate opt-in native MLX adapter. All
[32 ordinary/marked outputs](evidence/wide-mlx-2026-09-28/README.md) completed;
10,239 sampled tokens reconcile with exact native bytes, returned text and draw
journals. The 248,320-token head uses an ordered sparse projection with tested
reference arithmetic, without enlarging the frozen engine's bounds. Runtime EOS
248046 is explicitly checked; nested model metadata instead names 248044.

This path requires the content-pinned `mlx-community/Qwen3.5-9B-4bit` snapshot
`8b2b98c00a6b4d291155e4890773ca8f769aee53`, MLX 0.32.2, MLX-LM 0.31.2 and
Transformers 5.16.1. It is available through the unpublished Python API only:

```python
from keyprint import Keyprint

with Keyprint.from_mlx(
    "/path/to/pinned-snapshot",
    key=Keyprint.new_key(),
    execution="experimental-wide",
) as wm:
    result = wm.generate("Write a short explanation of a rainbow.", max_tokens=192)
    print(result.text)
```

Its tokenizer normalizes input to NFC; generated output bytes are never
normalized or repaired. Literal inspection rejects text changed by tokenization.
Existing reference/default behavior stays unchanged. No research acceptance,
detector threshold, provider-client qualification or Qwen-family-wide support
transfers. The paired study's semantic ratings are pending. JSON grammar, tools,
reasoning, streaming, batching and concurrent native serving are unqualified.
The [eight upstream ordinary baselines](evidence/model-baseline-2026-09-28/README.md)
remain separate evidence and include factual failures.

## Other application SDKs

| Client layer | Current Keyprint evidence | Next validation needed |
| --- | --- | --- |
| OpenAI Python | Local text and typed JSON; versions 1.109.1 and 3.14.1 with direct clients; 3.16.2 via LangChain; 3.20.0 via PydanticAI | Any additional API features or version combinations |
| Anthropic Python | Local text and typed JSON; versions 0.83.0 and 1.6.0 | Any additional API features or version combinations |
| LangChain ChatOpenAI | 1.6.2, langchain-core 1.6.4; sync and async text, exact replay, pre-inference tool/multi-turn rejection | Other backends, agents, tools, streaming and structured-output wrappers |
| Ollama Python | 0.6.2: generate, chat, async chat, exact replay through Keyprint llama.cpp worker | Native Ollama daemon integration and other backend/client combinations |
| PydanticAI | 2.51.0 on Qwen/MLX: sync/async text, explicit NativeOutput JSON, exact replay and request-ID conflict tested | Tools, system prompts, history and streaming unsupported; other backends untested |
| LiteLLM | 1.102.0 Router: two explicit local routes, sync/async, replay and timeout recovery tested | Proxy, automatic failover, distributed replay and additional runtimes |
| Vercel AI SDK | Not tested with Keyprint | Non-streaming request shape, headers, response parsing and retry behavior |

## New LangChain evidence

Three ordinary and three marked Llama 3.2 3B Q8_0 responses were generated on
macOS ARM64 CPU through llama-cpp-python 0.3.35. English and Spanish used
`invoke`; French used `ainvoke`. Each marked request was replayed with identical
text and usage, with exactly three marked model calls and three journals total.
Tools and multi-turn requests returned HTTP 400 before additional inference.
Returned text and completion counts matched private receipts.

[Plan](evidence/langchain-2026-09-21/plan.json) and
[all outputs](evidence/langchain-2026-09-21/results.json) are retained, including
unrequested elaboration in the Spanish marked answer. These are integration
samples, not a quality acceptance, causal comparison or detector calibration.
No hosted provider calls were made; LangSmith tracing was disabled.

[Runnable example](examples/langchain_local.py). Reproduce from an environment
with the server, llama-cpp and LangChain dependencies installed:

```sh
python tools/validate_langchain.py --model /path/to/Llama-3.2-3B-Instruct-Q8_0.gguf \
  --output private-langchain-check
```

The output directory must not already exist. Keep its keys and journals private;
only its `public/` files are intended for sharing. Model hash and exact dependency
versions are recorded before inference. This check runs locally; CI remains disabled.

## Why the distinctions matter

PydanticAI now has [actual-inference evidence](evidence/pydantic-ai-2026-09-28/README.md)
and an [optional recipe](examples/pydantic_ai_local.py). Eight outputs and 216
committed tokens reconcile with private reports; native JSON preserves the tested
names, deadline and approval condition. Replayed requests and rejected features
consume no additional inference. This adds no PydanticAI dependency to Keyprint
and does not establish general quality, detector or agent-loop support.

[LangChain](https://docs.langchain.com/oss/python/integrations/chat/openai),
[vLLM](https://docs.vllm.ai/en/latest/serving/online_serving/openai_compatible_server/)
and [Ollama](https://docs.ollama.com/api/openai-compatibility) document
OpenAI-compatible interfaces. That describes the request protocol, not installation
of Keyprint's sampler. SGLang and vLLM require separate native integration and
lifecycle testing. [PydanticAI](https://pydantic.dev/docs/ai/models/openai/) supports
custom endpoints; the scoped local path is tested above. Additional features and
backends still need their own end-to-end checks.

See the [detailed runtime audit](INTEGRATIONS.md),
[output-quality contract](OUTPUT_QUALITY.md) and [release gates](RELEASE_GOAL.md).


## September 21 native serving extension

vLLM 0.29.0+cpu now passes actual OpenAI 3.14.1 and Anthropic 1.6.0 HTTP
requests on the pinned CPU image, with the custom Keyprint processor installed
server-side. Four simultaneous ordinary/marked requests and a recovery request
returned 99 native token IDs matching journals exactly. Two Anthropic responses
returned 50 output tokens whose text and counts match selected-token receipts;
that protocol does not independently return final token IDs. All seven full
responses are retained. A streamed request was disconnected after one received
token; native sampling stopped after two selections, before the 384-token cap,
and another request succeeded. Graceful shutdown exited 0 without OOM.

This closes the previously untested native HTTP/client pilot and one disconnect
case. It does not establish exhaustive streaming equivalence, durable replay,
GPU execution, sustained load or production readiness. Native vLLM does not
inherit the Keyprint preview server's idempotency/cancellation extension.

[Results](evidence/vllm-server-2026-09-21/results.json) ·
[Independent token/text audit](evidence/vllm-server-2026-09-21/audit.json) ·
[Native server setup](tools/VLLM_SERVING.md).

The first native-server command used the wrong class separator and failed at
startup. Client attempt 1 used removed Anthropic Python sampling keyword
arguments; attempt 2 sent only x-api-key to a bearer-authenticated vLLM server.
The working client explicitly sends the local bearer token and local sampling
settings via `extra_body`. Failed runs and their returned outputs are retained.

## Additional ecosystem candidates, without growing the core

- [LM Studio](https://lmstudio.ai/docs/developer/openai-compat): local API server;
  no Keyprint native hook is implemented. Reusing an HTTP protocol is insufficient.
- [Bifrost](https://github.com/maximhq/bifrost): external gateway/router candidate;
  no Keyprint end-to-end test yet. Apply the same route identity and retry rules.
- [LiteLLM](https://docs.litellm.ai/docs/routing): tested explicit-route recipe;
  its automatic fallback machinery requires separate validation.
- PydanticAI: tested optional text/native-JSON recipe; additional agent features
  remain unsupported. Vercel AI SDK still needs its own actual request test.

SGLang remains at its earlier modified CPU hook pilot. Its runtime image/build
cache is no longer present locally, so no new lifecycle pass is claimed. Restore
the pinned build and measure HTTP traffic, disconnect/cancellation, reuse and
shutdown before promoting it. No hosted CI or notifications were enabled.
