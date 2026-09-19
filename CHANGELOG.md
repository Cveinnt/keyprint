# Changelog

## 0.1.0a1 (unreleased)

- Keep accepted server and playground work alive after a cancelled HTTP handler.
  Recover completed results without duplicate generation; drain accepted work
  during graceful shutdown and reconnect the playground after a refresh.
- Add a real OpenAI-client timeout recovery test with local inference to CI.
- Remove full-vocabulary Python arithmetic on excluded logits and zero weights
  in the portable Transformers and experimental serving adapters. Preserve
  represented probabilities, exact token selection and random-draw transcripts;
  record the new execution source separately. The frozen MLX engine is unchanged.
- Show actual playground stages and separate generation/inspection timings.
  Reuse the final prefix measurement instead of scoring the same text again.
- Add a packaged local playground: real paired generations, custom prompts,
  editable text, prefix signal curves, an independent-key control and JSON
  reports. Restore the previous live run on refresh without generating again.
- Add typed `Keyprint.inspect()` results while retaining the raw diagnostic
  report. Unavailable measurements remain unavailable, never zero or a verdict.
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
