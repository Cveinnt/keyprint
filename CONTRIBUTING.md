# Contributing

The launch is on hold. Keep fixes on a branch and preserve evidence provenance.

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
