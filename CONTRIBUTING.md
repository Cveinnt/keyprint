# Contributing

The launch is on hold. Keep fixes on a branch and preserve evidence provenance.

Use Python 3.12 or 3.13 and a separate virtual environment. Install
`pip install '.[test]'` for base tests, or `pip install '.[test,transformers]'`
for the optional runner tests. Run `python -m pytest tests` from the checkout.
Build with `python -m build`; test the resulting installed wheel outside the
source tree. CI keeps the old reference suite separate from the new package.

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
