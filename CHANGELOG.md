# Changelog

## 0.1.0a1 (unreleased)

- Add the `keyprint` distribution, `keyprint` Python namespace and readable CLI.
- Add key creation, local generation, installation diagnostics and source checks.
- Port reference imports into a private namespace without changing `sys.path`.
  Preserve old source hashes, record new ones, and compare reference behavior.
- Add an experimental Transformers CPU backend with explicit ByteLevel token
  bindings. Keep its profile and empirical status separate from the reference.
- Preserve the pinned MLX/Qwen backend and private durable failure reports.

This version is not on PyPI. Social launch remains postponed. SGLang/vLLM,
OpenAI-client serving, hosted-provider workflow and broad quality validation
are not complete. Historical 22/25 scoped acceptances are not a new release pass.
