# Keyprint

**Follow a text watermark through the choices a model makes.**

[Play with the demo](https://keyprint.vercel.app/#experiment) · [Compare 144 real output pairs](https://keyprint.vercel.app/#outputs) · [Read the investigation](https://keyprint.vercel.app/research/)

Keyprint is an independent research candidate built from Anthropic's public
watermark description. Explore the mechanism, inspect the sampler, and reproduce
the recorded quality arithmetic. It does not identify Anthropic's private
implementation or detect arbitrary Claude text.

## Run your first comparison

Python 3.12+ is required. Install the [PyPI prerelease](https://pypi.org/project/keyprint-research-v3/0.0.4rc3/) from a downloaded copy of this repository:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install keyprint-research-v3==0.0.4rc3
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

For real prose, start with the [recorded output viewer](https://keyprint.vercel.app/#outputs).
Model integration is a separate step: see [the full SDK API guide](sdk/README.md#supplied-model-caller-and-journal).

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
