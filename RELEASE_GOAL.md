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

The fixed predictability-filter development screen recovered two short-answer
misses: 10/12 marked hits overall and 3/5 in the short range, with zero ordinary
or wrong-key hits across the 24 opened outputs. Two short answers still miss.
An initial variable-batch numerical audit failed; the fixed-batch rerun retained
the same detection counts and passed all 48 prefix stability checks with exact
model-head identity. This adds an 8B-model inference dependency and has
not established a deployment false-positive rate or low-cost detection. The
candidate stays outside the SDK until independent validation supports it.

The predictability-null development run finished all 500 attempts, but only 499
produced usable controls: eight were flagged and one failed during selection
receipt writing because the disk was full. The run remains incomplete, its
IID-only bound is null, and the independent audit correctly rejects it. The
failed control is not a negative and no detector requirement closes from this run.
The separate partial audit verified all 998 surviving text/key scores, preserving
the original failed audit. Among the same 499 controls, the candidate adds five
false flags and removes two baseline flags. Its earlier short-answer recovery
therefore comes with an observed false-flag tradeoff. No SDK promotion or
threshold adjustment is justified by this opened-data experiment.

Storage cleanup enabled the September 20 paired Qwen serving experiment:
48 measured requests completed, with 2,009 journal-verified tokens including
warmups. Incremental marked/ordinary time per token was 1.0763, with a one-sided
95% bootstrap upper ratio of 1.0802. This fails the predeclared 5% timing screen.
All measured outputs reached EOS, but eight French outputs missed the exact
time-format screen. All 24 text pairs remain available for review. A18 and
quality gates stay open; profile the marking path before another declared
timing run, preserving this result and avoiding an after-the-fact threshold change.
Actual profiling subsequently retained twelve additional outputs (512 tokens)
and identified dense softmax iteration in both arms plus HMAC/tournament work
in the marked arm. A development-only HMAC-context prototype preserved 540
synthetic comparisons and 14,280 bits from actual committed-token contexts.
Its narrow timing improvement does not close A18 or change SDK execution.
The subsequent explicit experimental supplied-head pipeline has a separate
runtime identity. Its freshly installed wheel matched twelve actual Qwen outputs
and all 494 model steps against the reference, including probability hashes,
randomness transcripts, text and counts. It is not integrated into default
generation or serving by default. The subsequent reusable caller is available
only with `execution="experimental-fast"`; its own reporting contract binds the
experimental sources. The fresh caller wheel passes 290 tests, complete-caller
parity across 24 actual outputs/494 tokens per path, and real HTTP cancellation,
terminal replay and worker reuse. Production lifecycle and new paired timings
remain open. All frozen reference files and the prior failed timing result
remain unchanged.

The follow-up interleaved experiment completed 96 measured requests with no
errors or capped outputs. Its independent audit reconciled 4,028 tokens including
four warmups. Optimized execution reduced seconds per token by about 43% versus
the concurrently measured reference, but its marked/ordinary ratio was 1.0700
with a one-sided 95% upper ratio of 1.0744. The unchanged 5% incremental screen
still fails. All 48 text pairs remain available; sixteen French exact-time
format flags remain. This closes a measurement task, not A18 or quality.

The SHA-context follow-up preserves 24 real outputs/494 tokens per execution
path, passes actual HTTP cancellation/replay/reuse and 353 installed-wheel tests.
The separate 96-request serving repeat verifies 4,102 tokens including warmups.
Optimized marked/ordinary cost is 1.0607 (one-sided 95% upper 1.0684), still above
the unchanged 5% screen. Marked execution uses 0.5580 of the concurrently measured
reference time per token, but sixteen French format flags and broader quality,
detection and serving acceptance gaps remain. The NumPy tournament candidate is
not integrated because small-support timings regressed. Profile-guided progress
does not close the remaining production gates.

The batched-tournament implementation keeps scalar arithmetic below 64 eligible
candidates and preserves exact layer values/counters above that boundary. It
passes 388 installed-wheel tests, 24 real output comparisons, independent replay
of 988 committed tokens and actual HTTP cancellation/replay/reuse. Its separate
96-request timing run verifies 4,100 tokens including warmups and is the first
local 5% development-screen pass: marked/ordinary ratio 1.04354, one-sided 95%
upper 1.0498404. The narrow pass does not close A18 or justify launch. Freeze the
candidate before independent confirmation, wider workloads and separate startup
qualification. Sixteen French exact-time flags remain; quality, reader
indistinguishability, short-text detection and hosted release checks remain open.

The independently declared Dolly timing workload did not confirm readiness.
All 64 requests were retained: 29 EOS, 34 capped outputs and one ordinary request
that raised UnicodeDecodeError when its token cap ended inside a character.
Exact replay matched 192 original model heads and 302 recorded draws; pending
UTF-8 bytes were e2 9c. Its timing analysis remains null and the auditor rejects
the incomplete study. Fix graceful character-boundary finalization while retaining
all sampled tokens, raw trailing bytes and consumed-work receipts, then declare
a fresh confirmation. Do not convert this failed attempt into a success or treat
the earlier development timing pass as production acceptance.

The experimental caller now finalizes an exact token limit with a valid UTF-8
prefix and explicit pending-byte receipt. Complete output, EOS and control
boundaries retain strict decoding. The reproduced 192-token failure passes on
the separately identified fixed runtime with the same 192 model-head hashes,
302 draws and consumed-work counts; no extra token is sampled. Literal replay
of that incomplete carrier is unavailable. The fresh installed wheel passes
415 tests, including the OpenAI-client length response and idempotent replay;
actual Qwen-over-HTTP cancellation, terminal replay and worker reuse also pass.
The frozen reference/default caller remains unchanged. The original failed
confirmation remains failed, and the new runtime needs fresh timing confirmation.
This repairs a concrete experimental-path blocker without closing a release gate.

The post-repair repetition completed all 64 measured requests and sixteen warmups
without errors. Independent byte/draw reconciliation verified 10,727 committed
tokens and 16,441 draws. Twenty-eight measured outputs were capped; all original
texts are retained. The unchanged serving screen still fails: marked/ordinary
seconds per token 1.053402, one-sided 95% upper 1.057928. This is a repetition of
the same workload, not unseen-task validation. The original failed study stays
failed. A post-hoc source review of 24 outputs also identifies classification
contradictions and unsupported additions in both paths. Quality and A18 remain
open. Profile the failed workload before proposing another execution change.

GitHub CI for `67799b3` and `e0f1f6b` did not start. Both workflows failed before executing
steps, with GitHub reporting failed account payments or an Actions spending
limit. This is a runner-account blocker, not a passing check or a diagnosed code
failure. The last successful hosted revision is `7510b96`. Local validation
continues; hosted validation
must be rerun after account access is restored. Do not change billing settings
or public-launch status as a workaround.

## First-use acceptance

At the user's September 20 request, both hosted CI workflows are manually
disabled to stop pre-execution failure notifications. Continue with local
installed-wheel and actual-inference checks; do not re-enable or repeatedly
dispatch hosted CI without a new user instruction.

The subsequent local ARM Docker SGLang run completed all six ordinary/marked
pairs and three callback contract checks. Independent verification matches all
661 returned tokens, condition labels and pinned runtime sources. The verifier
now also rejects stale or missing Keyprint native-adapter and sampling-source
identities, with sixteen focused tests; a synthetic before/after case confirms
the previous verifier accepted a wrong adapter hash. This does not authenticate
untrusted callers or substitute source hashes for execution evidence. Four
mechanical screening failures remain: ordinary email, both French responses and
marked JSON. A post-hoc assistant review also finds semantic problems missed by
literal screens, including an incorrect ordinary scattering explanation and
poor Spanish in both conditions. These twelve outputs are compatibility evidence
for the tiny SmolLM2 fixture, not production-language quality acceptance.

The portable Transformers path now handles a token cap inside a UTF-8 character
by retaining every committed token and pending byte, returning the valid prefix
and explicitly marking full-carrier replay unavailable. Complete output still
requires exact tokenizer rendering; invalid bytes and partial EOS remain strict.
The fresh wheel passes 487 Python tests and seven JavaScript tests. Real pinned
SmolLM2 ordinary and marked Japanese generations both exposed incomplete first
characters, then passed exact one-token prefix replay with matching model-head
hashes, random draws and commits and no extra model call. Real OpenAI/Anthropic
client generation and replay also pass. These are bounded reliability checks;
the generated Japanese samples are not quality acceptance. Default frozen MLX
and native SGLang/vLLM cap handling remain separate open work.

The September 20 fresh-wheel browser pass completed the default Qwen/MLX
prefilled pair, a custom prompt and edited-text inspection. It verified refresh
recovery, keyboard scrubbing and a 390-pixel mobile layout. Actual Unicode output
exposed a browser UTF-16/Python code-point mismatch: the final chart prefix was
shorter than the generated text. Chart coordinates, prefix slicing and the
half-text control now use the SDK's code-point units. Undoing an edit also clears
the stale pending-measurement label. Three regression tests fail on the previous
source and pass on the repair, and are included in provider CI. The repaired
browser pass verifies every prefix of a real emoji-containing response and a
longer edited-text axis. Deliberately dropping a completed HTTP response then
retrying replays the same request ID and exact result without a second attempt.
The initial SVG assertion used an unsupported Playwright innerText operation;
that harness failure is retained separately from the corrected textContent pass.
These scoped checks do not establish detection, output quality, broad browser
support or the complete first-use/release gate.

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

The local server now accepts the Anthropic Messages text subset alongside OpenAI
Chat Completions. Both endpoints share the worker, request budget, cancellation
and process-lifetime idempotency; cross-protocol reuse of a key is rejected.
A fresh installed wheel passes 461 tests. Actual OpenAI 3.14.1 and Anthropic 1.6.0
requests on pinned Qwen/MLX experimental execution and SmolLM2/Transformers CPU
match private generated-text and usage receipts and replay without generation.
Actual Anthropic-client cancellation on both backends preserves consumed work,
replays the terminal error and succeeds on a subsequent 32-token request.
The initial SmolLM2 validator assumed the MLX report envelope; that harness
failure and its outputs were retained before normalization and a fresh check.
This advances client compatibility; tools, streaming, multiple turns, other
model bindings, native hosted sampling and production acceptance remain open.

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
