# Keyprint

**Text watermarking you can inspect, generate and test locally.**

Private development preview. The launch is postponed while integrations and
claims are audited. This branch builds **`keyprint 0.1.0a1`**; that name/version
has not been published to PyPI. The old `keyprint-research-v3` release is a
separate research reference, not the install command for this branch.

## Start here

Python 3.12 or 3.13. The [interactive playground](#the-interactive-playground)
is the main demo: generate, edit, and inspect real text with a local model.
For a model-free installation check first, run from this checkout:

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install .
keyprint demo
```

```text
Ordinary  BDABCAAD
Marked    CDAABDCB
```

This offline fixture shows how a private-key weighting step changes token
selection. It needs no account, model download or API key. The example uses
public demonstration randomness and A/B/C/D choices; it does not generate prose.

`keyprint doctor` checks installation imports and source integrity.
`keyprint verify` checks the namespaced engine against its manifest. Neither
checks model compatibility or detector accuracy. Add `--json` for full reports.

## The interactive playground

After installing a backend and downloading its pinned model below:

```sh
pip install '.[server]'
keyprint playground
```

Open the local session URL printed in your terminal. On Apple Silicon the
default backend is MLX; elsewhere it is Transformers. The command finds the
documented pinned model in the Hugging Face cache. For models downloaded to
another directory, use `keyprint playground --backend transformers --model
models/smollm2` or `keyprint playground --backend mlx --model models/qwen3-8b-4bit`.
Missing assets produce an actionable error, never an implicit download.

The prefilled prompt runs two real generations on first load. Refreshing restores
the previous live run from this process instead of generating again. If work is
still running, refresh restores its prompt and response limit immediately and
waits for its result. Those controls stay locked while work runs. A failed attempt
stays failed until you explicitly start another experiment. Enter your
own prompt, compare ordinary and marked responses, edit the marked text, and
inspect the changing signal alongside an independent-key control. The chart
recomputes literal diagnostics at text prefixes. Its fractions are observed bit
counts, **not confidence percentages or calibrated detection**. Both responses
use independent randomness; wording differences are not a quality experiment.
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

## What works, and what does not

| Stack | Scope |
| --- | --- |
| Python / NumPy | Supplied-logit reference pipeline and offline demo |
| MLX | Exact pinned Qwen3-8B-4bit model on Apple Silicon |
| Transformers | Experimental local CPU float32 text generation; SmolLM2 integration tested |
| vLLM | Experimental CPU 0.29.0 adapter: two batched SmolLM2 generations; not a production integration |
| SGLang | Experimental pinned ARM CPU source build: six ordinary/marked SmolLM2 pairs with returned-token verification; NUMA workaround required, quality unvalidated |
| OpenAI Python client | Real local HTTP request tested; single-message Chat Completions subset |
| OpenAI-hosted GPT / Anthropic-hosted Claude | Their public APIs do not expose this custom sampler hook; no native integration |
| Completed GPT / Claude prose | Explicit experimental local rewrite; original retained, meaning and detection unvalidated |

See [provider examples](PROVIDERS.md) for `keyprint serve` and
`watermark.rewrite_openai(response)` / `watermark.rewrite_anthropic(message)`.
These helpers never imply that a hosted provider ran the Keyprint sampler.

The normal portable backend runs one response at a time. Streaming, batching, tools,
reasoning channels, beam search, speculative decoding, quantized checkpoints
and grammar constraints are unsupported. An AI setup wizard would not solve
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
