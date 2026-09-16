# Keyprint

[![SDK checks](https://github.com/Cveinnt/keyprint/actions/workflows/test.yml/badge.svg)](https://github.com/Cveinnt/keyprint/actions/workflows/test.yml)

**Follow a text watermark through the choices a model makes.**

[Play with the demo](https://keyprint.vercel.app/#experiment) · [Compare 144 real output pairs](https://keyprint.vercel.app/#outputs) · [Read the investigation](https://keyprint.vercel.app/research/)

Keyprint is an independent research candidate built from Anthropic's public
watermark description. Explore the mechanism, inspect the sampler, and reproduce
the recorded quality arithmetic. It does not identify Anthropic's private
implementation or detect arbitrary Claude text.

## Run your first comparison

Python 3.12+ is required. Clone this repository and install the [PyPI prerelease](https://pypi.org/project/keyprint-research-v3/0.0.4rc3/):

```sh
git clone https://github.com/Cveinnt/keyprint.git
cd keyprint
python3.12 -m venv .venv
.venv/bin/python -m pip install keyprint-research-v3==0.0.4rc3
.venv/bin/python examples/doctor.py
.venv/bin/python examples/compare.py
```

On Windows, use `py -3.12 -m venv .venv`, then `.venv\Scripts\python.exe`
instead of `.venv/bin/python`. Windows runtime validation is not yet recorded.

Expected fixture output:

```text
ordinary: 'BDABCAAD' (generation_trace)
marked  : 'CDAABDCB' (generation_trace)
```

This eight-step A/B/C/D illustration uses supplied logits, a public key and
repeatable seeded draws. It loads no model, requires no API key, and does not
generate prose. Both conditions start from the same supplied heads and draw
stream; changing the weights can change which choices are selected. The saved
`fixture-comparison.json` retains both complete reports and their interpretations.
Choose a fresh `--output` path to run it again. A positive or negative diagnostic
is not an authorship verdict.

For existing prose examples, use the [recorded output viewer](https://keyprint.vercel.app/#outputs).

## Generate your own prose locally

The runnable [MLX example](examples/generate_mlx.py) supports one pinned
Qwen3-8B-4bit checkpoint on Apple Silicon macOS. It takes your prompt, creates
a fresh private key, and generates ordinary and marked responses. Model weights
are several GB; the explicit download below runs once. Generation stays local.

From the cloned repository and environment above:

```sh
.venv/bin/python -m pip install mlx==0.32.2 mlx-lm==0.31.2 transformers==5.16.1
.venv/bin/python examples/doctor.py --mlx
.venv/bin/hf download mlx-community/Qwen3-8B-4bit \
  --revision 545dc4251c05440727734bcd94334791f6ab0192 \
  --local-dir models/qwen3-8b-4bit
.venv/bin/python examples/generate_mlx.py \
  --model models/qwen3-8b-4bit \
  --prompt "Explain why the sky is blue in two short sentences." \
  --max-tokens 64
```

The example checks checkpoint and tokenizer hashes before loading. It refuses
other models instead of silently using the wrong vocabulary. Already downloaded
this exact revision? Pass its local directory to `--model` and skip the download.

Outputs, full reports, the private key and durable journals go to
`private-keyprint-run/` (owner-only directory). **Do not publish the key or
journals.** Use a fresh `--output private-keyprint-run-2` for another attempt.
Failures are retained, never silently retried. Each condition uses independent
random draws, so wording differences cannot be attributed solely to watermarking.
The default 64-token cap can truncate a response. Neither timing nor two outputs
establishes serving overhead, semantic equivalence, or detection accuracy.

These examples live in GitHub and use the unchanged PyPI release. For application
integration, see [the model caller contract](sdk/README.md#supplied-model-caller-and-journal).

## Integration coverage

| Stack | Current support |
| --- | --- |
| Plain Python / NumPy | Installed SDK and supplied-logit fixtures; no model needed |
| MLX + pinned Qwen3-8B-4bit | Runnable local generation example above |
| OpenAI / Anthropic hosted APIs | No adapter; this sampler needs access before token selection, which their public generation APIs do not expose |
| Transformers / vLLM | No supported adapter yet; model access alone does not establish tokenizer or sampler compatibility |
| LangChain / LlamaIndex | No wrapper; orchestration cannot supply the missing sampling access |

An AI setup wizard is not required. `examples/doctor.py` checks package presence,
dependency conflicts and bundle integrity without changing your environment or
loading a model. It does not certify model compatibility. CI builds and tests an
installed wheel on Linux and macOS with Python 3.12 and 3.13; model generation is
a separate local check. Windows is not a validated runtime for this release.

## Inspect or change the implementation

The `sdk/` tree comes from the published 0.0.4rc3 source archive. Install it in a
separate environment if you want to change code:

```sh
python3.12 -m venv .source-venv
.source-venv/bin/python -m pip install ./sdk
.source-venv/bin/python -m unittest discover -s sdk/tests
.source-venv/bin/python -m keyprint_v3 verify
```

- [Public API exports](sdk/keyprint_v3/__init__.py) and [CLI](sdk/keyprint_v3/__main__.py)
- [Reporting facade](sdk/keyprint_v3/_bundle/research/keyprint_v3_public_api_rc2.py)
- [Exact categorical sampler](sdk/keyprint_v3/_bundle/research/keyprint_exact_categorical_v2.py)
- [Post-filter support policy](sdk/keyprint_v3/_bundle/research/keyprint_stable_support_filter_v3.py)
- [Supported API tests](sdk/tests/test_public_surface.py)
- [Release notes](sdk/RELEASE_NOTES.md) and [third-party notices](sdk/THIRD_PARTY_NOTICES.md)

The historical paths inside `_bundle` preserve immutable evidence and are not
supported import paths. Use `keyprint_v3` in application code. `verify` checks
bundled file hashes, not authorship, provenance signatures or detector accuracy.

## What the evidence establishes

The [25-requirement ledger](https://keyprint.vercel.app/clue-ledger.json) has
22 scoped acceptances. Output quality, reader indistinguishability and serving
overhead remain open. These are local tests derived from the public description,
not 25 independent promises from Anthropic. The evidence does not establish a
unique reconstruction, universal domain coverage or deployment error rates.

[Recompute the published quality arithmetic](https://keyprint.vercel.app/reproduce/README.md)
without downloading a model. This replay verifies stored inputs and arithmetic;
it does not independently rerate the outputs or reproduce model generation.

## Help test the limits

Useful contributions include minimal numerical counterexamples, independent
domain evaluations and clearer explanations. Include version, platform, a
minimal reproducer and the complete interpretation from any report. See
[CONTRIBUTING.md](CONTRIBUTING.md). Please exclude private keys and private text.

MIT project code by Vincent Wu (Cveinnt). Upstream tokenizer/data terms remain
separate. No affiliation with Anthropic.
