# Compatibility at a glance

Updated September 21, 2026. Applies to the **private, unpublished Keyprint
0.1.0a1 development branch**. “Tested” means actual inference through the
listed local path. It does not mean production qualification, arbitrary model
support or transfer of the original research acceptances.

| Runtime | Keyprint path | Tested model / environment | OpenAI Python | Anthropic Python | LangChain | Limits |
| --- | --- | --- | --- | --- | --- | --- |
| MLX | Tested local | Qwen3-8B · Apple Silicon | Tested | Tested | Not tested | Single response; typed JSON tested. Production batching unqualified. |
| Transformers | Tested local | SmolLM2 / SmolLM3 · CPU | Tested | Tested | Not tested | Single response; typed JSON tested. Other tokenizer families unqualified. |
| llama.cpp / GGUF | Tested local | SmolLM2 / Llama 3.2 3B · CPU | Tested | Tested | Tested on Llama | LangChain sync + async; no tools, streaming or constrained JSON. |
| SGLang | Experimental hook | Pinned modified ARM CPU build · SmolLM2 | Not tested | Not tested | Not tested | 620 returned tokens matched journals. Server lifecycle and GPU unqualified. |
| vLLM | Experimental hook | 0.29.0+cpu · SmolLM2 | Not tested | Not tested | Not tested | 125 batched tokens matched journals. Server lifecycle and GPU unqualified. |
| Ollama | No Keyprint adapter | Upstream OpenAI-compatible API | Not integrated | Not integrated | Not integrated | An API-compatible endpoint does not expose the required native sampling hook. |
| Hosted GPT / Claude | No Keyprint sampling hook | Provider-hosted generation | Client only | Client only | Not integrated | Local client compatibility does not watermark hosted GPT or Claude. Rewriting is blocked. |

Both provider clients connect to **Keyprint's local model**. They do not add
Keyprint to hosted GPT/Claude generation. Supported local requests contain one
user text message, a token cap and an idempotency key. Tools, system/multi-turn
messages and streaming are rejected before inference. Replay lasts only for
the server process. See [provider usage](PROVIDERS.md) for exact boundaries.

## Other application SDKs

| Client layer | Current Keyprint evidence | Next validation needed |
| --- | --- | --- |
| OpenAI Python | Local text and typed JSON; versions 1.109.1 and 3.14.1 with direct clients; 3.16.2 via LangChain | Any additional API features or version combinations |
| Anthropic Python | Local text and typed JSON; versions 0.83.0 and 1.6.0 | Any additional API features or version combinations |
| LangChain ChatOpenAI | 1.6.2, langchain-core 1.6.4; sync and async text, exact replay, pre-inference tool/multi-turn rejection | Other backends, agents, tools, streaming and structured-output wrappers |
| PydanticAI | Not tested; a configurable OpenAI endpoint alone is insufficient | Explicit Chat Completions route, request headers, parsing, replay and rejected features |
| LiteLLM | Not tested with Keyprint | Proxy routing, retries, request transformation and response accounting |
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

[LangChain](https://docs.langchain.com/oss/python/integrations/chat/openai),
[vLLM](https://docs.vllm.ai/en/latest/serving/online_serving/openai_compatible_server/)
and [Ollama](https://docs.ollama.com/api/openai-compatibility) document
OpenAI-compatible interfaces. That describes the request protocol, not installation
of Keyprint's sampler. SGLang and vLLM require separate native integration and
lifecycle testing. [PydanticAI](https://pydantic.dev/docs/ai/models/openai/) supports
custom endpoints; it still needs its own end-to-end Keyprint check.

See the [detailed runtime audit](INTEGRATIONS.md),
[output-quality contract](OUTPUT_QUALITY.md) and [release gates](RELEASE_GOAL.md).
