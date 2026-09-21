# Keyprint

**Text watermarking you can inspect, generate and test locally.**

Private development preview. The launch is postponed while integrations and
claims are audited. This branch builds **`keyprint 0.1.0a1`**; that name/version
has not been published to PyPI. The old `keyprint-research-v3` release is a
separate research reference, not the install command for this branch.

[Framework and client compatibility](COMPATIBILITY.md): tested local OpenAI,
Anthropic and LangChain requests; separate experimental SGLang/vLLM hooks.

## Start here

The interactive playground generates two real responses, lets you change the
prompt, and shows how edits affect the watermark signal. No hosted API key or
account is needed. Start from this private checkout with Python 3.12 or 3.13.

**Apple Silicon Mac:**

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install '.[mlx,server]'
keyprint playground --download
```

**CPU alternative:** replace the install and launch commands with
`python -m pip install '.[transformers,server]'` and
`keyprint playground --backend transformers --download`. This uses the small
SmolLM2 integration example; its answer quality is limited.

Open the complete local session URL printed in the terminal. The prefilled
example runs once when the model is ready. Then enter your own prompt, compare
the responses, or edit and inspect the marked text. Automatic rewriting of
existing text is blocked until language and meaning preservation are validated.
Results are real local model output;
the signal chart is an uncalibrated diagnostic, not a detection verdict.

The explicit `--download` flag fetches only the pinned model files into your
Hugging Face cache: about [4.62 GB for Qwen/MLX](https://huggingface.co/mlx-community/Qwen3-8B-4bit/tree/545dc4251c05440727734bcd94334791f6ab0192)
or [270 MB for SmolLM2](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/tree/12fd25f77366fa6b3b4b768ec3050bf629380bac).
Runtime dependencies and inference memory are additional. Downloads reuse cached
files and never install Python packages, request credentials or run remote model
code. On later runs, `keyprint playground` uses the cache without downloading.
Use `--model PATH` for existing local files; do not combine it with `--download`.
If a download fails, the cache is retained and the demo does not start. Retry
explicitly when ready. Windows/Linux execution remains subject to the
[tested compatibility scope](INTEGRATIONS.md), not inferred from CLI routing.

For an optional model-free check: `python -m pip install .`, then `keyprint demo`.
That fixture uses A/B/C/D choices and does not generate prose. `keyprint doctor`
checks package imports and engine integrity; `keyprint verify` checks the engine
manifest. Neither certifies a model or detector. Add `--json` for reports.

## The interactive playground

With a backend installed and pinned assets cached, check setup and start offline:

```sh
keyprint doctor --playground
keyprint playground
```

Open the local session URL printed in your terminal. On Apple Silicon the
default backend is MLX; elsewhere it is Transformers. The command finds the
documented pinned model in the Hugging Face cache. For models downloaded to
another directory, use `keyprint playground --backend transformers --model
models/smollm2` or `keyprint playground --backend mlx --model models/qwen3-8b-4bit`.
Missing assets produce an actionable error; only `--download` authorizes a fetch.

`doctor --playground` checks backend and server imports before starting a
session. For MLX it verifies pinned model-file hashes; for Transformers it
checks metadata syntax and the presence of weights. It does not allocate a
model or certify compatibility. Use the same `--backend` and `--model` options
as the playground; a successful check prints the exact next command.

On qualified Apple Silicon installations, the reviewed optional native wheel
can be selected explicitly:

```sh
keyprint doctor --playground --execution experimental-native
keyprint playground --execution experimental-native
```

Install the matching wheel as described in [native setup](native/README.md)
first. The same option works with `generate` and `serve`. The default stays
`reference`; native execution remains experimental and does not pass the
complete serving-cost target. This option is MLX-only and never installs or
downloads an accelerator implicitly.

Open the printed URL while the model loads. The page shows the real startup
state and elapsed time; you can edit the prompt before generation begins.
Generation stays disabled until the model is ready. Startup failures leave the
page available and record details in the private `startup.json`; fix the model
path or dependencies and restart the command. **Check connection** reconnects
to the server without reloading a model or silently retrying a failed experiment.

The prefilled prompt runs two real generations on first load. Refreshing restores
the previous live run from this process instead of generating again. If work is
still running, refresh restores its prompt and response limit immediately and
waits for its result. Those controls stay locked while work runs. A failed attempt
stays failed until you explicitly start another experiment. Enter your
own prompt, compare ordinary and marked responses, edit the marked text, and
inspect the changing signal alongside an independent-key control. The chart
recomputes literal diagnostics at text prefixes. Its fractions are observed bit
counts, **not confidence percentages or calibrated detection**. Prompt-mode
responses use independent randomness; wording differences are not a quality experiment.
Reading view formats basic headings, lists, bold text and code blocks. Switch to
Exact text to see every original character. Formatting never changes the text
used for editing, inspection or export; generated HTML, links and images remain
inert text. Long response panels are keyboard-focusable and scrollable.
Measured edits also survive a refresh within the same server process. A failed
inspection restores its input alongside the last successful measurement, clearly
labeled as stale. A new completed generation clears the previous pair's edit.
While work runs, the page shows the actual generation or inspection stage and
elapsed time. Each result separates generation from inspection time. These are
individual local observations, not a serving-throughput benchmark.
Use **Stop** to end an active generation or inspection at the next safe boundary.
Controls stay locked until the worker stops; completed responses and measurements
remain visible. Refresh reconnects to the same attempt, including a pending stop,
without starting new work. An active model step or inspection must finish first.

**Automatic rewriting is unavailable.** `rewrite()`, `rewrite_openai()` and
`rewrite_anthropic()` raise `RewriteUnavailableError` before inference or output
creation. The playground disables rewriting and rejects direct rewrite requests.
Known translations and changed conditions are failures, not acceptable watermark
outputs. Keep existing text unchanged. Use `generate()` only for new responses;
it is not an equivalent way to watermark an existing document. Native generation
still needs output-quality evaluation. Historical rewrite samples remain in the
[quality review](QUALITY_REVIEW.md); their lexical checks do not certify meaning.
See [sampling performance](PERFORMANCE.md) for reproducible arithmetic and
real-model parity checks, with the remaining performance limits.
See [actual inference testing](INFERENCE_TESTING.md) for paired text comparisons,
real SDK-over-HTTP checks, CI artifacts and observed quality failures.

Keys stay in the local Python process. A fresh key is saved in the owner-only
session directory unless `--key PATH` supplies an existing key. The independent
control key is also retained privately for reproduction. Prompts, reports
and private generation journals stay in that directory. The browser receives no
watermark key or journal. Keep the session URL private. This is a bounded local preview,
not a public production server; closing the tab does not cancel an in-flight run.

## Generate real text

`generate`, `serve` and `playground` share the same default: MLX on Apple
Silicon macOS, Transformers elsewhere. `--backend` overrides that choice.
After downloading the documented pinned model into the Hugging Face cache,
you can omit `--model`:

```sh
keyprint generate --key keyprint.key --prompt 'Explain why the sky is blue.'
```

For another local directory or the larger SmolLM3 model, keep `--model PATH`.
An empty cache reports the exact pinned download command and starts no download.
Backend selection is a convenience, not a claim that every operating system or
model is qualified. The compatibility matrix remains authoritative.
Blank prompts, prompts over 16,000 characters and response caps outside 1–1,024
tokens are rejected before loading model weights. Malformed schema JSON is also
rejected before model allocation; schema support is checked during generation.

The Transformers backend runs locally on CPU. Download the explicit model
revision once, then generation requires no network access:

```sh
pip install '.[transformers]'
hf download HuggingFaceTB/SmolLM2-135M-Instruct \
  --revision 12fd25f77366fa6b3b4b768ec3050bf629380bac \
  --include '*.json' '*.safetensors' '*.jinja' \
  --local-dir models/smollm2
keyprint keygen
keyprint generate --backend transformers --model models/smollm2 \
  --key keyprint.key --prompt 'Explain why the sky is blue.'
```

The command prints text and the location of its private report. It records
failures without retrying them. `--max-tokens 128` changes the response cap;
`--condition ordinary` generates the unmarked control under the same base
filter. Different runs use independent randomness, so wording changes alone
are not evidence of a watermark's effect on quality.

SmolLM2 is a small integration example, not a quality benchmark or recommended
production model. The portable profile supports explicit ByteLevel BPE token
bindings and rejects unsupported tokenizers. Its evidence is separate from
the pinned Qwen research profile. See [integration coverage](INTEGRATIONS.md).

### Local GGUF with llama.cpp

The optional CPU backend runs without Torch or MLX. From this reviewed checkout:

```sh
pip install '.[llama-cpp,server]'
keyprint playground --backend llama-cpp --download
```

The explicit download fetches a pinned 145 MB SmolLM2 Q8_0 integration fixture.
If a compatible upstream wheel is unavailable, installing `llama-cpp-python`
requires a C/C++ build toolchain. This is an experimental adapter, not a claim
of production answer quality. For an existing local GGUF:

```python
from keyprint import Keyprint

with Keyprint.from_llama_cpp("model.gguf", key=Keyprint.new_key()) as model:
    result = model.generate("Explain why the sky is blue.")
    print(result.text)
```

Use `--backend llama-cpp --model model.gguf` with `generate`, `serve` or
`playground`. Only decoder-only, non-recurrent GPT-2 byte-BPE bindings are
admitted. The measured fixture and platform are listed in
[integration coverage](INTEGRATIONS.md#local-gguf-with-llamacpp).
GPU execution, Ollama, streaming, tools, JSON constraints and other model
families are not qualified by this backend. The native context handles one
request at a time and rejects concurrent use; the context manager releases it.

A second measured binding is Llama 3.2 3B Instruct Q8_0 (3.42 GB). Download it
explicitly, under the model's Llama 3.2 license, then use the same interface:

```sh
hf download bartowski/Llama-3.2-3B-Instruct-GGUF Llama-3.2-3B-Instruct-Q8_0.gguf \
  --revision 5ab33fa94d1d04e903623ae72c95d1696f09f9e8 --local-dir models/llama3.2
keyprint playground --backend llama-cpp \
  --model models/llama3.2/Llama-3.2-3B-Instruct-Q8_0.gguf
```

Twenty fresh ordinary/marked outputs completed on this binding. Factual errors,
invented details and rewrite meaning drift remain visible in the
[quality review](QUALITY_REVIEW.md). A larger model does not establish production
quality or transfer detection evidence from another model.

To try a larger multilingual model, the same CPU adapter accepts the pinned
SmolLM3-3B tokenizer and requests its non-thinking chat template:

```sh
hf download HuggingFaceTB/SmolLM3-3B \
  --revision a07cc9a04f16550a088caea529712d1d335b0ac1 \
  --include '*.json' '*.safetensors' '*.jinja' \
  --local-dir models/smollm3
keyprint generate --backend transformers --model models/smollm3 \
  --key keyprint.key --prompt 'Explain why the sky is blue in two sentences.'
```

Weights download is approximately 6.2 GB; CPU float32 weights alone use about
12.3 GB, with additional memory required for inference. This remains an
experimental integration, not a production-quality or detector guarantee.
`enable_thinking=False` is passed to the model's template; templates that ignore
that option are not thereby qualified for non-thinking output. No reasoning or
tool channel is parsed, and generated text is never silently stripped.

### JSON that clients can parse

For MLX or Transformers, install the optional `structured` extra and pass
`json_schema` to `generate`. Keyprint masks invalid tokens before sampling; it
does not remove Markdown fences, repair the answer or retry behind the scenes.

```sh
pip install '.[transformers,structured]'
keyprint generate --backend transformers --model models/smollm3 \
  --key keyprint.key --prompt 'Return a JSON record for Maya, count 3, enabled false.' \
  --json-schema record.schema.json --max-tokens 128
```

`record.schema.json`:

```json
{"type":"object","properties":{"name":{"type":"string"},"count":{"type":"integer"},"enabled":{"type":"boolean"}},"required":["name","count","enabled"],"additionalProperties":false}
```

Python: `result = kp.generate(prompt, json_schema=schema, max_tokens=128)`.
Check `result.report["structured_output"]["schema_validated"]` before consuming
the JSON. A token cap can leave an incomplete document. Completed responses are
checked independently against the schema; grammar/validation failures retain
private receipts and raise `KeyprintError`.

The local server also supports typed OpenAI and Anthropic parsing helpers;
see [provider examples and schema limits](PROVIDERS.md#typed-json-output).
The same `json_schema` argument works with `Keyprint.from_mlx(...)`; install
`.[mlx,structured]` for the pinned Qwen backend. Native SGLang/vLLM structured
generation is not integrated. This mode's constrained
distribution has no calibrated detector claim; a fixed schema can leave little
or no room for a watermark. Valid JSON does not establish factual correctness.

## Python API

```python
from pathlib import Path
from keyprint import Keyprint

watermark = Keyprint.from_transformers(
    "models/smollm2",
    key=Path("keyprint.key").read_bytes(),
)
result = watermark.generate("Explain why the sky is blue.")
print(result.text)
print(result.artifacts)  # private journal and report
inspection = watermark.inspect(result.text)
print(inspection.fraction)  # observed one-bit fraction, not confidence
```

Reuse the same private key to inspect matching-key diagnostics with
`watermark.inspect(result.text)`. The typed result exposes `events`, `ones`,
`trials`, `fraction` and the unchanged scientific `report`. An optional
`key=other_key` inspects the same text with another key as a control.
`watermark.score(result.text)` remains available for the raw report.
These are **uncalibrated diagnostics**, without
an authorship verdict, detection threshold or false-positive guarantee.
Retokenizing visible text can differ from the generated token path.

To stop generation cooperatively, pass `cancel_event=stop`, where `stop` is a
`threading.Event`, and call `stop.set()` from another thread. Catch
`KeyprintCancelled` to inspect retained work and receipts. Keep the model on its
owning thread; an active model call cannot be preempted. The local server also
provides [explicit cancellation](PROVIDERS.md#explicit-cancellation), separate
from HTTP timeouts and result recovery.

If tokenized input exceeds the backend's limit, catch `InputLimitError` (a
`ValueError`) and inspect `input_tokens`, `limit`, and `max_tokens`. Counts include
the chat template and any rewrite instructions. `max_tokens=None` means an
input-only limit; otherwise the response budget also consumes context. Shorten
the input or, for a context limit, lower the response budget. Rejected inputs
do not start sampling. Both local client endpoints and the playground return
HTTP 400 with these counts; retrying the same request ID replays that rejection.
Use a new request ID after changing the input or budget.

Keys are exactly 32 bytes. `keyprint keygen` creates an owner-only file without
printing the key or overwriting an existing file. `Keyprint.new_key()` returns
fresh bytes for applications; store them securely yourself. Keep keys and
journals private. Losing the key prevents later matching-key inspection.

For Apple Silicon, the reference backend remains available:

```sh
pip install '.[mlx]'
hf download mlx-community/Qwen3-8B-4bit \
  --revision 545dc4251c05440727734bcd94334791f6ab0192 \
  --local-dir models/qwen3-8b-4bit
keyprint generate --backend mlx --model models/qwen3-8b-4bit \
  --key keyprint.key --prompt 'Explain why the sky is blue.'
```

Python uses `Keyprint.from_mlx("models/qwen3-8b-4bit", key=...)`. This backend
checks the exact model/tokenizer asset hashes. Other MLX models are rejected.

An opt-in execution candidate is available with
`Keyprint.from_mlx("models/qwen3-8b-4bit", key=..., execution="experimental-fast")`.
It reuses a bound candidate and preserves durable generation/cancellation
receipts with a distinct experimental runtime and reporting schema. The default
uses reference sampling with separately identified token-limit finalization.
See [performance evidence](PERFORMANCE.md#separately-identified-experimental-execution)
for the validation scope; this option does not establish a serving-speed claim.

A separate, unpublished `keyprint-native` wheel enables
`execution="experimental-native"` on Apple Silicon macOS. It batches the same
HMAC computation and bounded top-k selection in a bundled library, preserving
exact reference tie-breaking. This SDK requires accelerator version `0.1.0a2`
and rejects older wheels before model loading; the core package never compiles or
downloads it automatically. Full-caller output parity and both local provider
cancellation paths have been tested, with a distinct binary-bound identity.
Validated unkeyed metadata is prepared once per candidate; each request still
checks pinned asset bytes and binding fields and owns fresh sampler state.
Native v2 reports use [versioned lossless vector commitments](tools/VECTOR_COMMITMENTS.md);
default and reference reports keep their legacy dense hash fields.
See [native installation and scope](native/README.md). It is optional, not the
default execution or a production-performance guarantee.

On MLX generation paths, a token limit inside a UTF-8 character returns the
valid text prefix with `completion="length"`. The report's `carrier_rendering`
retains every committed token ID and the unfinished bytes as `pending_utf8_hex`.
That carrier's literal-score diagnostic is unavailable; the rendered prefix is
not presented as the complete sampled text. EOS, channel boundaries and invalid
byte sequences remain strict. Default MLX reports use the
`keyprint.bounded-reference-report.v1` schema. The archived research engine and
its strict finalizer remain unchanged; old results retain their original runtime
identities and do not automatically qualify the repaired caller.

## What works, and what does not

| Stack | Scope |
| --- | --- |
| Python / NumPy | Supplied-logit reference pipeline and offline demo |
| MLX | Exact pinned Qwen3-8B-4bit model on Apple Silicon |
| Transformers | Experimental local CPU float32 text generation; SmolLM2 and SmolLM3-3B integration tested; output quality unqualified |
| vLLM | Experimental CPU 0.29.0 adapter: two batched SmolLM2 generations; not a production integration |
| SGLang | Experimental pinned ARM CPU source build: six ordinary/marked SmolLM2 pairs with returned-token verification; NUMA workaround required, quality unvalidated |
| OpenAI Python client | Real local HTTP request tested; single-message Chat Completions subset |
| Anthropic Python client | Real local HTTP requests on pinned Qwen/MLX and SmolLM2/Transformers; one user string or text block, explicit token cap |
| OpenAI-hosted GPT / Anthropic-hosted Claude | Their public APIs do not expose this custom sampler hook; no native integration |
| Completed GPT / Claude prose | Rewriting blocked before inference; language and meaning preservation unvalidated |

See [provider examples](PROVIDERS.md) for `keyprint serve` and the rewrite
restriction. Local client compatibility does not insert Keyprint into hosted
OpenAI or Claude generation.
The optional `clients` extra accepts OpenAI `>=1.109.1,<4` and Anthropic
`>=0.83.0,<2`; both older and newer client pairs have local inference evidence.
See the provider guide for exact tested versions and limitations.

The portable Transformers backend runs one response at a time. Streaming,
batching, tools, reasoning channels, beam search, speculative decoding and
quantized checkpoints are unsupported. Optional bounded JSON-schema constraints
are supported through the `structured` extra, as described above; arbitrary
grammars are unsupported. An AI setup wizard would not solve
these compatibility gaps. Explicit extras, short commands and useful errors do.

## Research scope

Keyprint is an independent candidate inspired by Anthropic's public watermark
description. It does not identify their private implementation, detect arbitrary
Claude text, or prove that competing approaches cannot work.

The published reference ledger has **22 scoped acceptances out of 25**. Output
quality, reader indistinguishability and serving overhead remain open. Those
acceptances belong to the recorded reference configuration. They do not transfer
automatically to this namespaced port, portable profile or another model.

## Development and provenance

```sh
pip install '.[test,transformers,server,clients]'
python -m pytest
```

- [`src/keyprint`](src/keyprint): supported Python API and adapters.
- [`sdk`](sdk): preserved prior reference distribution and tests.
- [`port-manifest.json`](src/keyprint/_engine/port-manifest.json): original and
  ported engine hashes. Import rewrites create a new execution identity.
- [`tools/port_reference.py`](tools/port_reference.py): reproducible namespace
  transformation; refuses to overwrite an existing port.
- [Contributing](CONTRIBUTING.md) and [integration audit](INTEGRATIONS.md).

Tests compare the port against the preserved reference in separate interpreters,
check private artifacts and failure behavior, and reject unsupported tokenizer
bindings. Real-model checks are recorded separately from fixture tests.

MIT. Copyright Vincent Wu (Cveinnt). See [LICENSE](LICENSE) and
[third-party notices](sdk/THIRD_PARTY_NOTICES.md).
