# Keyprint release goal

Deliver a focused, dependable watermarking library with Pretext-level ease of
use and an interactive demonstration that makes its real capability visible.
Publicity remains postponed until the release candidate meets this bar.

## First-use acceptance

- A developer can install the published package and follow one tested path to
  a useful result. Core, model assets and optional inference dependencies have
  separate, measured installation costs.
- A prefilled playground uses the actual SDK and model, accepts a custom prompt,
  compares ordinary and marked output, and lets the user edit and inspect text.
- Every visual derives from measured results. Show unavailable and weak signals,
  truncation, failures and wrong-key controls. No handpicked success presented
  as a live run; no decorative graphs presented as measurements.
- The public API stays small; detailed reports remain accessible when needed.
- Detection verdicts require calibration and held-out controls. Raw diagnostics
  may be demonstrated but cannot stand in for a production detector.

## Integration acceptance

Support is explicit by framework, version, model/tokenizer and hardware.
Qualify SGLang/vLLM, establish an Ollama/llama.cpp path, and exercise actual OpenAI
and Anthropic clients and applications. Protocol interoperability must remain
distinct from native sampling access to hosted GPT/Claude.

## Scientific and operational acceptance

Keep all 25 public-disclosure requirements visible with their evidence and
remaining gaps. Preserve failed experiments. Reference results do not transfer
automatically to new adapters or models. Quality, indistinguishability, detection
calibration and serving overhead need evidence for the claimed configuration.
Validate clean installs, concurrency, cancellation, retries and private-key
handling before describing the SDK as production-ready.

## Launch presentation

Lead with a useful capability, a reproducible interactive result, then a short
installation path. Put research depth and limitations one layer below the main
flow without hiding them. Keep the serif-led editorial visual direction, warm
neutral backgrounds, restrained ink/rust accents, and readable charts. Avoid
emoji, decorative dashboards and unsupported universal-compatibility claims.
