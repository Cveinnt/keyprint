# Contributing

Human and AI-assisted contributions are welcome. Pick a scoped task from
[the roadmap](ROADMAP.md), reproduce the current behavior, and submit the
smallest change with its evidence. Start with [AGENTS.md](AGENTS.md) when using
a coding agent. [Community guide](COMMUNITY.md) explains where to share work.

Good first contributions include a small demo built from exported comparison
data, a reproducible installation bug, or a targeted Unicode/EOS regression.
Research changes need fresh evaluations with declared criteria; changing a
threshold until an existing dataset passes is not qualification.

You remain responsible for reviewing and understanding an AI-assisted patch.
State which checks ran, which skipped, and which require a model. Retain failures
and original output. Do not paste private prompts, keys or raw journals into PRs.

Use Python 3.12 or 3.13 and a separate virtual environment. Install
`pip install '.[test]'` for base tests, or `pip install '.[test,transformers]'`
for the optional runner tests. Run `python -m pytest` from the checkout.
Default discovery covers `tests/`. The preserved `sdk/tests/` suite and native
framework probes under `tools/` run explicitly in their separate environments;
they are not silently collected into a base SDK installation.
Build with `python -m build`; test the resulting installed wheel outside the
source tree. Hosted SDK and SGLang workflows are manually disabled at the
owner's request. Run checks locally; do not re-enable hosted CI or substitute a
base suite with skipped integrations for the full release matrix.

For local server/client and JSON-contract coverage, install the `server`,
`clients`, `structured` and backend extras alongside `test`. Use a fresh
environment for each advertised backend so unrelated optional dependencies
cannot hide missing requirements. MLX runs on Apple Silicon; the Transformers
test path is CPU-only. Add the separately reviewed native wheel only when
qualifying `experimental-native`. Run browser-code checks with
`node --test tests/test_playground_ui.cjs tests/test_reader.cjs`.

Actual inference is separate from unit tests. Follow `INFERENCE_TESTING.md`,
retain ordinary/marked text and private journals, and reconcile outputs before
reporting compatibility. The first-use path is `keyprint playground --download`:
it explicitly fetches pinned public assets, then starts the local demo. Also
test `keyprint playground` with a populated cache and both `HF_HUB_OFFLINE=1`
and `TRANSFORMERS_OFFLINE=1`. Model
downloads and cached reuse are different checks; report which actually ran.

Do not edit `sdk/keyprint_v3/_bundle` or regenerate its manifest to make a test
pass. New model bindings and algorithms receive new identities and evidence.
The namespaced port manifest records original and changed hashes explicitly.

Each integration needs a documented model/tokenizer contract, actual model
execution, unsupported-mode rejection, Unicode/EOS handling, and tests of
failure and cache behavior. Server adapters additionally need request isolation,
batch reordering, cancellation and prefix-cache tests. A mocked callback is not
proof of a working server integration.

Do not commit keys, private journals, model weights, API credentials or personal
prompts. Use public fixtures for bug reports and publish only reviewed evidence.
A score is not an authorship judgment. Claims must point to the exact measured
configuration; preserve failures as well as successes.

## Client version checks

Keep these checks local while hosted CI is disabled. Test the built wheel in
separate fresh environments with both dependency sets:

| Set | OpenAI | Anthropic |
| --- | --- | --- |
| Lower bounds | 1.109.1 | 0.83.0 |
| Additional tested pair | 3.14.1 | 1.6.0 |

For each set, install the wheel's `test,server,clients,structured,mlx` extras
alongside the exact client versions above, then run `python -m pip check`.
Use the Transformers extra instead of MLX for a separate CPU backend check;
the version-matrix inference evidence currently covers MLX only.

Run the client contract tests against each installed wheel:

```sh
python -m pytest tests/test_rewrite.py tests/test_server.py \
  tests/test_messages_server.py tests/test_mlx_structured.py -q -rs
```

Then run `tools/validate_mlx_structured.py` and `tools/validate_cancellation.py`
with each client pair, the pinned Qwen model, `--execution reference`, and fresh
output directories. Cancellation needs separate `--protocol openai` and
`--protocol anthropic` runs with `--backend mlx`. Follow
[inference testing](INFERENCE_TESTING.md) for assets and receipt handling.
Do not treat fixture tests, a resolver pass, or accepted version ranges as
proof of real inference across every client release.

## llama.cpp local checks

CI remains disabled. Run these against an installed, reviewed wheel; use a
fresh output directory for every attempt. The measured macOS CPU build used
`CMAKE_ARGS='-DGGML_METAL=OFF -DGGML_BLAS=OFF -DGGML_OPENMP=OFF'` and
`CMAKE_BUILD_PARALLEL_LEVEL=2` while installing the `llama-cpp` extra.

```sh
python -m pytest tests/test_llama_cpp.py tests/test_server.py tests/test_playground.py
keyprint doctor --playground --backend llama-cpp --model /path/to/model.gguf
python tools/validate_compatibility.py --backend llama-cpp \
  --model /path/to/model.gguf --output gguf-results --http-client
python tools/validate_compatibility.py --backend llama-cpp \
  --model /path/to/model.gguf --cases tools/quality_cases.json --output gguf-longform
python tools/validate_cancellation.py --backend llama-cpp --protocol openai \
  --model /path/to/model.gguf --output gguf-cancel-openai
python tools/validate_cancellation.py --backend llama-cpp --protocol anthropic \
  --model /path/to/model.gguf --output gguf-cancel-anthropic
python tools/validate_portable_utf8.py --backend llama-cpp --condition ordinary \
  --model /path/to/model.gguf --output gguf-utf8-ordinary
python tools/validate_portable_utf8.py --backend llama-cpp --condition marked \
  --model /path/to/model.gguf --output gguf-utf8-marked
```

Read both generated texts in `gguf-results/public/comparison.html`. Engineering
success does not approve quality. Native libraries and the GGUF are hashed in
the reports; each tokenizer binding is separate from the reference profile.
The fixture download is explicit through `keyprint playground --backend
llama-cpp --download`. No hosted provider calls or model code downloads occur.

`--cases` accepts 1 to 64 bounded synthetic cases. Prompts and ordinary/marked
outputs are exported into the report; do not use private or customer content.
Safe unique IDs prevent artifact collisions. `action: rewrite` invokes the real
SDK rewrite path with the supplied source and `preserve` phrases. The report
records a canonical case hash. Explicit word limits use whitespace counts;
literal and rewrite checks do not determine meaning. Read the
[retained quality findings](QUALITY_REVIEW.md) before making quality claims.
