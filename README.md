# Keyprint

**Text watermarking you can inspect, generate and test locally.**

Private development preview. The launch is postponed while integrations and
claims are audited. This branch builds **`keyprint 0.1.0a1`**; that name/version
has not been published to PyPI. The old `keyprint-research-v3` release is a
separate research reference, not the install command for this branch.

## Start here

Python 3.12 or 3.13. From this checkout:

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
```

Reuse the same private key to inspect matching-key diagnostics with
`watermark.score(result.text)`. These are **uncalibrated diagnostics**, without
an authorship verdict, detection threshold or false-positive guarantee.
Retokenizing visible text can differ from the generated token path.

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
| SGLang | Experimental pinned ARM CPU source build: two batched SmolLM2 generations; NUMA workaround required |
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
python -m pytest tests
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
