# Changelog

## 0.1.0a1 (unreleased)

- Add the `keyprint` distribution, `keyprint` Python namespace and readable CLI.
- Add key creation, local generation, installation diagnostics and source checks.
- Port reference imports into a private namespace without changing `sys.path`.
  Preserve old source hashes, record new ones, and compare reference behavior.
- Add an experimental Transformers CPU backend with explicit ByteLevel token
  bindings. Keep its profile and empirical status separate from the reference.
- Preserve the pinned MLX/Qwen backend and private durable failure reports.
- Add `keyprint serve`: authenticated local Chat Completions subset with a
  single model worker, bounded attempts and process-lifetime idempotency.
- Add explicit local rewriting of completed OpenAI/Anthropic prose, retaining
  originals, candidates and failed checks. No automatic meaning/detection pass.
- Add experimental CPU serving adapters and separate pilot scripts. Integration
  receipts do not transfer reference research acceptances.

This version is not on PyPI. Social launch remains postponed. SGLang/vLLM,
production serving, hosted-provider product readiness and broad quality validation
are not complete. Historical 22/25 scoped acceptances are not a new release pass.
