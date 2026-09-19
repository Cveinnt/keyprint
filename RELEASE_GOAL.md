# Keyprint release goal

Deliver a focused, dependable watermarking library with Pretext-level ease of
use and an interactive demonstration that makes its real capability visible.
Publicity remains postponed until the release candidate meets this bar.
The user's September 19 direction makes production quality of both SDK and
demos the release objective. A green test suite alone does not satisfy it.

## Current release gates

- [ ] First-use flow: a fresh install reaches a useful, real result through one
  documented path; dependency/model costs and failures are understandable.
- [ ] Supported configurations: real inference, returned-token verification and
  supported client contracts pass on each advertised framework/version/model.
  Do not advertise an adapter based only on a fixture or earlier source version.
- [ ] Output quality: retained negation, factual, multilingual and constrained
  output failures are resolved or the affected feature is excluded explicitly
  from the supported release. Lexical checks cannot approve semantic fidelity.
- [ ] Detection: preregistered thresholds and held-out ordinary/wrong-key controls
  establish false-positive bounds and power for the claimed configurations.
- [ ] Serving reliability: bounded concurrency, cancellation, retries, cache
  reuse and private-key handling are exercised on actual supported inference.
- [ ] Demo quality: prefilled and custom-input paths work; results and provenance
  are visible; loading, error, weak-signal and recovery states are usable; keyboard
  and mobile flows pass browser checks. The main demo demonstrates the capability
  without presenting canned output or an uncalibrated fraction as a verdict.
- [ ] Release: exact wheel, docs, demo and compatibility matrix agree; final CI
  and clean-install checks pass; publication and announcement claims match evidence.

September 19 verification: revision `82ce577` passed all ten SDK CI jobs plus
the dedicated ARM SGLang actual-inference job. Cooperative cancellation
changes pass 208 tests against a freshly installed wheel. Actual SmolLM2 and
Qwen runs both stopped after one committed token, retained consumed-work counts,
replayed terminal cancellation without regeneration, and generated successfully
on the same worker. This does not preempt model kernels or qualify framework
cancellation, crash recovery or streaming. The browser playground now exposes
Stop for generation and inspection, preserves completed results, and reconnects
after refresh. Actual Qwen generation cancellation passed browser checks. A
disclosed pause after a real SDK inspection exercised inspection-stop races on
desktop and mobile without substituting measurements or generated text.
The `82ce577` SGLang repeat verified twelve outputs and 960 journal-matching
tokens, with five mechanical quality flags. Green CI verifies the integration
and receipt contract, not the generated answers' correctness.
CI still needs verification at the exact release revision. The latest frozen
Qwen weighted-reference screen completed all 24 outputs:
8/12 marked detections, no ordinary or wrong-key hits, and four truncated
outputs. Within the original 100–400-word control range it detected only 1/5
marked answers; longer marked answers were 7/7. The separate fresh 500-document
null screen had five false hits, an IID-only upper bound of 2.32%. Neither result
qualifies a production detector. The completed 10,000-passage WikiText screen
had 115 false hits (1.15%; IID-only 97.5% upper bound 1.38%). A learned marginal
weight development candidate did not improve marked detections and was not
promoted into the SDK. These are separate research results, not SDK verdicts.
Local paired runs also exist for Transformers and vLLM. SmolLM2 rewrites lost an
approval condition, and Qwen French changed exact time formatting. After fixing
disk capacity and verifying the SGLang runtime source, a comparison attempt
returned ten texts with 731 matching tokens before its 420-second timeout.
Both negation outputs dropped the backup instruction. Its three callback
contract checks passed, but the partial suite is not integration acceptance.
The subsequent complete SGLang run passed all six pairs and three contract
checks, with 620 matching returned tokens and verified condition labels. All
twelve outputs reached EOS. Factual, email, negation and multilingual failures
remain in those outputs; the CPU callback pass is not quality acceptance.
The independent ARM CI run returned twelve texts with 978 matching tokens,
including two truncated outputs and three mechanical screening failures.
Calibration, broad compatibility and production acceptance remain open. Keep
launch held.

The `d4d85e3` SDK CI passed all ten jobs after the browser Stop change; the local
fresh-wheel suite passed 211 tests. An exact replay of all five short marked
answers then verified 1,590 original model steps with no new random draws.
All five carry positive model-path marking evidence, but four remain missed by
the existing text-only rule. Tokenization matches exactly for those five.
The replay is a private-state diagnostic, not a replacement detector. Investigate
prompt-free predictability estimates next, with a separate frozen candidate and
fresh power/null evaluation before any claim changes.

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
