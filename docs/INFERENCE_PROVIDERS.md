# Try Keyprint in your inference stack

[Play without installing](https://keyprint.vercel.app/play/) · [Exact compatibility matrix](../COMPATIBILITY.md) · [Client protocol reference](../PROVIDERS.md)

Keyprint is an open research SDK for generation-time text watermarking. Start
with one supported local model, one request and its exact output. It is not yet
a production watermarking service or a drop-in adapter for every inference server.

## Choose your entry point

| You control… | Start with | Current boundary |
| --- | --- | --- |
| A Python model worker | `Keyprint.from_mlx`, `.from_transformers` or `.from_llama_cpp` | Specific model/tokenizer bindings; [measured scope](../INTEGRATIONS.md) |
| An application using OpenAI or Anthropic clients | The local trial below | Local Keyprint models, one user message, no tools or streaming |
| A vLLM deployment | [Native CPU pilot](../tools/VLLM_SERVING.md) | A real server-side sampler pilot; GPU/load qualification open |
| An SGLang deployment | [Native integration evidence and setup](../INTEGRATIONS.md) | Pinned modified ARM CPU runtime; production lifecycle/GPU open |
| Ollama or LM Studio's native daemon | [Compatibility boundaries](../COMPATIBILITY.md) | No native Keyprint sampler adapter yet |
| Hosted GPT or Claude calls | Use the recorded demo to explore the mechanism | Their client SDKs do not expose the native Keyprint sampling hook |

Protocol compatibility and native sampling integration are different. A custom
base URL points a client at Keyprint's model; it does not install Keyprint inside
an arbitrary server. Router/client recipes for [LangChain](../examples/langchain_local.py),
[LiteLLM](../examples/litellm_local.py), [Ollama Python](../examples/ollama_local.py)
and [PydanticAI](../examples/pydantic_ai_local.py) are optional and do not enlarge
the core package dependencies.

## First local request

Use a cloned repository and Python 3.12 or 3.13. The example below assumes the
[pinned SmolLM2 files](../USAGE.md#generate-real-text) already exist at `models/smollm2`.
This small CPU model is an integration fixture, not a quality benchmark.

```sh
python -m pip install '.[server,clients,transformers]'
keyprint keygen keyprint.key
keyprint keygen local-api.key
keyprint serve --backend transformers --model models/smollm2 \
  --key keyprint.key --api-key local-api.key --output private-keyprint-server
```

Use fresh key/output paths for a new trial. For the measured Apple Silicon path,
install `.[server,clients,mlx]`, choose `--backend mlx` and pass your pinned
Qwen3-8B model directory. Model weights are additional to SDK dependencies.

In a second terminal, from the same directory/environment:

```python
from pathlib import Path
from uuid import uuid4
from openai import OpenAI

request_id = str(uuid4())
print("Request ID:", request_id)  # Retain for recovery of this attempt.
with OpenAI(
    base_url="http://127.0.0.1:8765/v1",
    api_key=Path("local-api.key").read_bytes().hex(),
    max_retries=0,
    timeout=120,
) as client:
    response = client.chat.completions.create(
        model="keyprint",
        messages=[{"role": "user", "content": "Explain why the sky is blue."}],
        max_completion_tokens=128,
        extra_headers={"Idempotency-Key": request_id},
    )
    print(response.choices[0].message.content)
```

The watermark key stays in the worker; only the separate API token goes to the
client. If a request times out, preserve its ID and body and use the same live
server process to recover it. Rerunning this example generates a new ID and is
a new attempt. Restarting the server loses its replay state. Full behavior and
the equivalent Anthropic example are in [PROVIDERS.md](../PROVIDERS.md).

## Make something other people can try

A local worker can create a recorded viewer with no model required for visitors:

```python
from keyprint import Keyprint

with Keyprint.from_mlx("path/to/qwen3-8b-4bit", key=Keyprint.new_key()) as wm:
    comparison = wm.compare("Explain why the sky is blue in two sentences.")
    comparison.export("my-demo")
```

Serve with `python -m http.server --directory my-demo`. Review prompts and
original outputs before sharing. `comparison.to_dict()` exposes the same data
for custom visualizations; [gallery and model-comparison recipes](../examples/README.md)
show other starting points. Recorded playback is different from live inference.

## What a production integration still needs

- A specific model/tokenizer/EOS contract and exact token-to-text reconciliation.
- Lifecycle evidence for that serving engine: batching, request isolation,
  cancellation, caching and recovery. The local preview is a single-worker subset.
- Quality and language evaluation on the provider's actual task distribution,
  retaining failures and comparing ordinary versus marked output.
- Detection calibration and fresh confirmation under declared false-positive
  criteria. Displayed fractions are not production detection confidence.

Do not translate, rewrite or repair completed text to make a marking experiment
look successful. Generation-time choices still need semantic evaluation.
The [25-requirement ledger](https://keyprint.vercel.app/#clue-conformance) records
historical, scoped reference results, not acceptance of every adapter or this
entire SDK. [Current research gaps](../ROADMAP.md) remain explicit.

To qualify another runtime, open a [bounded integration issue](https://github.com/Cveinnt/keyprint/issues/new?template=research.md)
with model/runtime versions, the native sampling hook, intended serving features
and a reproducible test plan. Contributions can use [AGENTS.md](../AGENTS.md).
