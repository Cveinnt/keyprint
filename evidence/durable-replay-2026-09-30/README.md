# Durable serving: implementation awaiting tests

This change addresses an SDK service failure mode: a restart previously erased
idempotency state, permitting a repeated request to generate again. The new
single-host SQLite journal reserves requests before model work, stores exact
terminal response bytes/status/headers, and treats interrupted reservations as
terminal uncertainty rather than retry authorization. Directory ownership,
model/key/server binding, response checksums, persistent request limits and disk
failure handling are implemented. No external dependency or sampler change.

**No new behavioral tests have run.** The 512 MiB watchdog refused startup because
macOS reported elevated system memory pressure. The refusal is retained in
`test-resource.json`. Syntax checks and source hashes appear in `validation.json`;
they do not establish durability, client compatibility or production readiness.

Twenty-one new restart/failure/ownership/corruption cases are written, including
an abrupt subprocess exit. Pending validation runs them alongside the existing
OpenAI, Anthropic and Ollama HTTP suites. Model-bound tests and a fresh wheel
check remain separate. No model loaded, unrelated application stopped, CI
triggered, site deployed, package published or launch post dispatched.

Implemented behavior and platform limits: [provider guide](../../PROVIDERS.md).
Scientific quality, detection and the full launch goal remain unresolved.
