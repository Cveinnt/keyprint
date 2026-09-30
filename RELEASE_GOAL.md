# Keyprint release goal

## Priority: watermark fidelity before launch polish

Pretext is the benchmark for launch impact, clarity, creative demonstrations and
ease of use. Its framework architecture, feature list and package size are not
requirements for Keyprint. Never trade watermark validity, language, meaning,
facts, conditions or public-clue conformance for a smaller or more viral demo.

Keep existing input text unchanged. The supported mechanism chooses tokens during
native generation; it does not translate, rewrite or silently repair completed
text. Its ordinary and marked samples need not be word-for-word identical, but
changed facts, intent, language or important details fail the preservation goal.
Do not present equivalent-looking wording as proof that meaning is preserved.

Validate task fidelity and watermark performance separately on declared profiles:
retained ordinary/marked outputs, fixed-key behavior, and detection power at the
registered false-positive target. A higher diagnostic fraction, selected good
key, altered threshold or favorable subset cannot substitute for those checks.
Public-source clue coverage remains evidence-backed; it does not identify
Anthropic's undisclosed implementation or establish untested guarantees.

The website size reduction only deferred loading unchanged comparison
records. The sampling engine, model bindings, tokenization and detector code
were untouched. The wording highlighter changes presentation only; generation
text and exported data remain exact. These changes provide no new scientific
acceptance and do not resolve existing quality or detection gaps.

## September 29 bounded-candidate evaluation: no promotion

The separately named paced research candidate has completed 128 EOS outputs and
24,421 audited token commits. All assistant source-only ratings were frozen
before joining conditions, keys or signal counts. Strict full-task passes are
25/64 ordinary and 16/64 marked; accepting every preflagged ambiguous judgment
gives 36/64 and 34/64. Every response stayed English in this English-only cohort.
Both arms contain factual errors. These descriptive results do not establish
noninferiority, human acceptance or preservation across languages and domains.
See [complete review](evidence/paced-quality-2026-09-29/README.md).

The uniform token-path score was then checked against 199 fresh random keys,
using a fixed conservative rank cutoff of 0.01. Only 1/64 marked and 0/64 ordinary
outputs cleared it. All 256 saved owner/control scores reproduced, and all 128
ranks were independently recomputed. This opened-cohort diagnostic does not
establish useful detection or a production false-positive rate. See
[fresh-key screen](evidence/paced-key-rank-2026-09-29/README.md).

The source passages were model prompts; generation used general-purpose mode,
with no exact-copy source protection active in this cohort. The candidate's
source-preservation machinery must not be mistaken for tested semantic fidelity
on these summarization tasks. Half-to-double probability bounds also do not
guarantee unchanged meaning.

The original study's separate native replay now verifies all 128 paths and
24,465 steps, retaining earlier interrupted attempts. This closes mechanical
replay, not quality. [Complete replay](evidence/source-replay-complete-2026-09-29/README.md).
Both model jobs finished below 7 GiB under the external memory watchdog, with
MLX cache disabled and cleanup verified. [Memory policy](MEMORY_SAFETY.md).

The saved-probability diagnosis now reconciles all 128 paths and 24,421 steps.
Median marked likelihood ratio is approximately 1.52:1; zero marked paths reach
the fixed 100:1 diagnostic cutoff. Mean accumulated conditional information is
0.534 nats per marked path. This points to limited per-response information, but
is not a population bound or an impossibility result. The privileged-probability
oracle is not an accessible text detector. See
[conditional information](evidence/paced-information-2026-09-29/README.md).
A complement-symmetric allocation comparison is also complete. It preserves the
same exact bounds and ideal-label conditional mean; a one-round integer proposal
now matches all forty numerical fixtures and 1,200 updates. The initial two
numerical failures remain retained. Uniform-score lift fell in every fixture
family, and highly peaked distributions gained little information, so this
variant is not promoted and does not justify a new model run. See
[allocation comparison](evidence/complement-paced-2026-09-29/README.md).
The probability-regime audit now covers all 24,421 saved positions: 27.64% of
marked positions carry 92.34% of measured conditional information, while 98.38%
of frozen layer calls occur in lower-information positions. The earlier
prompt-free filter improved its small development set but its larger null run
remains incomplete, with eight flags versus five on the same 499 usable controls.
Neither finding qualifies a detector. See
[regime audit and historical reconciliation](evidence/paced-regimes-2026-09-29/README.md).
Two exact aggregate-score rules have now been compared on every saved marked
prefix. Centered linear allocation reduces distortion and signal. Balanced
half-mass allocation yields 2.23 times expected score lift and 5.49 times
conditional information, but 2.68 times summed probability movement. Exact
normalization, excluded mass and [p/2,3p/2] bounds hold; none guarantee semantics.
Thirty tests pass. See [bounded score comparison](evidence/centered-score-2026-09-29/README.md).
The balanced rule now has a separate experimental profile, exact sampling
session, inference loop and token-path score replay. Thirty-eight new integration
checks pass; 135 combined new and regression checks pass. A guarded English/Spanish
native preflight stopped on global memory pressure at 5.33 GiB before any token
committed. Cleanup was verified and the interrupted attempt is retained without
automatic restart. See [integration and interruption](evidence/balanced-session-2026-09-29/README.md).
The subsequent guarded native preflight completed four prefixes and 64 tokens,
with zero execution or sampler-audit errors. The full study runner and verifier
are implemented. Fresh multilingual inputs are frozen: sixteen
English/Spanish/French/Chinese cases, 96 facts, four fresh
owner keys, 199 fresh decoys, 128 paired attempts and review/scoring criteria.
Sixteen input-integrity tests pass. See
[frozen inputs](evidence/balanced-evaluation-inputs-2026-09-29/README.md).
The study and one separately audited continuation now retain 71 complete outputs,
two infrastructure interruptions and 55 unstarted attempts. The 71 texts plus
the original sealed failure have source-only assistant ratings; the all-128
rating freeze and condition/score analysis remain incomplete. Both workers
stopped on global memory pressure below their 8 GiB cutoff, with cleanup verified
and no automatic restart. The latest continuation preserved every original
output and added eight. See [continuation evidence](evidence/balanced-continuation-2026-09-29/README.md).
This closes evidence-preserving continuation work, not quality or detection
acceptance. A second continuation is not supported yet and requires its own
validation before any new model load.
Any new confirmation needs prospectively frozen, disjoint material; do not
rerate or tune this development cohort into a pass. SDK defaults, approved site,
CI-off policy and publication hold remain unchanged.

## September 28 source-grounded execution and frozen factual review

- Completed all 128 predetermined English source-based outputs across sixteen
  tasks and four existing keys. Every output reaches EOS; zero runtime errors or
  truncations. No retries, rewriting or selection of favorable responses.
- All 24,465 native tokens pass the complete prompt/byte/usage/EOS/draw/commit
  receipt audit. Every blinded view matches its exact source, frozen rubric and
  recorded output. This is not independent model-forward replay or calibration.
- All 128 source-only assistant judgments and sixty uncertain fields were frozen
  before condition/key/count joins. Strict content passes are 33/64 ordinary and
  23/64 marked; accepting all pre-flagged ambiguities gives 43/64 and 37/64.
  Full-task passes are 32/64 and 22/64, or 42/64 and 36/64 under sensitivity.
- Required source-fact coverage passes 47/64 ordinary versus 36/64 marked;
  supported-claim checks pass 39/64 versus 38/64. Both arms retain English in all
  64 outputs and format in 63/64. Missing detail drives more of the measured
  difference than unsupported claims. This fixed 16-task/four-key study is
  descriptive, not a causal estimate or powered quality acceptance.
- All matching/next-key inspections are available, but correlated raw counts
  have no calibrated new-profile detector threshold. Independent arithmetic
  recomputation matches both strict and sensitivity component totals.
- Actual-data review UI passes desktop/mobile source visibility, blank ratings,
  format checks, navigation, persistence and partial export checks, with no
  console errors or overflow. QA actions are not human ratings.
- [Execution and review evidence](evidence/source-grounded-run-2026-09-28/README.md).
  SDK/defaults and landing design unchanged; no new scientific clue acceptance,
  package publication, deployment, hosted CI enablement or public launch.

## September 28 Qwen3.5 semantic review and full native replay

- Froze source-fact ratings on all 32 shuffled outputs before revealing
  conditions, key slots or raw counts. Strict content/full-task passes are 2/16
  ordinary and 0/16 marked. Accepting all seven conservative flags yields 6/16
  versus 3/16 content and 3/16 versus 2/16 full task. Format passes are 9/16 in
  each arm; language passes 16/16 ordinary and 15/16 marked.
- Clear failures include invented weekdays/logistics/deployment procedures, a
  marked 20-GB-versus-5-GB comparison called fivefold, and marked Japanese with
  the English word "itself". No wholesale translation. All judgments and failures
  retained. Ordinary failures create a floor effect, not proof of causal
  watermark harm or a new perfect-baseline acceptance gate.
- Independently replayed all 32 complete native paths: all 10,239 model-head
  hashes, transformed-weight hashes, categorical draws and committed tokens
  match. Direct MLX calls, gap-first full-head filtering and scalar reference
  source policy bypass SDK generation, wide projection and sparse source code.
  Model kernels, tokenizer binding and reference primitives remain shared.
- [Review, sensitivity and replay evidence](evidence/wide-mlx-2026-09-28/README.md).
  Fifty-four local research/checker regression tests pass. No SDK/default/engine
  change, hidden text repair, new package publication, deployment or CI enablement.
- [Next source-grounded selection](evidence/source-grounded-selection-2026-09-28/README.md):
  16 real English summarization contexts selected from a pinned Dolly snapshot
  by fixed hash, independent of reference answers or model outputs. Sources are
  private and separate from the MIT SDK. Task coverage criteria and the complete
  runner were subsequently frozen before the execution recorded above. This adds
  evidence alongside, rather than replacing, failed
  stress tests. No new-profile detector threshold, scientific clue or launch
  acceptance is claimed. Publicity stays held.

## September 28 native Qwen3.5 paired execution

- Added an explicit `execution="experimental-wide"` Python API for the pinned
  Qwen3.5-9B 4-bit MLX snapshot. Separate NFC/wider-head profile and runtime stop
  policy; default reference behavior, frozen numerical bounds and engine files
  remain intact. No silent output normalization or literal-inspection repair.
- Completed all 32 scheduled ordinary/marked generations on the unchanged four
  long tasks and all four existing keys. Zero execution errors. All 10,239 tokens
  reconcile with exact returned bytes, prompt IDs, EOS and 32 distinct draw
  transcripts. This receipt audit is not independent model-forward replay.
- Exact filter/probability/draw tests cover high token IDs, ties, extreme
  temperatures, subnormals, invalid padding and unchanged default rejection.
  Request-local cache and real stop-token tests pass. 291 focused local checks
  pass across the inference and Torch environments; three optional checks skip.
- Built and isolated-installed a wheel; all 78 Python source files match the
  recorded inference source, and all 39 frozen engine files verify. This is not
  a clean dependency-resolution or public registry publication claim.
- [Complete samples, receipts and blank review page](evidence/wide-mlx-2026-09-28/README.md).
  The subsequent review and native replay are recorded above. New-profile
  detector calibration and serving qualification remain open; no historical
  scientific clue closes. Public launch remains held.

## September 28 newer-model baseline

- Ran all eight planned ordinary Qwen3.5-9B MLX 4-bit responses on the unchanged
  four longer factual tasks. All reach EOS and retain language. Strict content
  and full-task passes: 2/8. Accepting both conservative judgments gives 4/8
  content and 3/8 full task. No watermark applied or post-generation repair.
- All 2,391 sampled tokens reconcile with upstream native decoding and retained
  events; 2,407 model forwards counted. All 13 model assets verified before
  inference. Seven new receipt checks and 35 combined local study checks pass.
- This establishes that facts can fail without watermarking; it does not
  excuse watermark-added harm or make the different-model comparison causal.
  [All outputs, frozen labels and methods](evidence/model-baseline-2026-09-28/README.md).
- Identified concrete Qwen3.5 admission issues: 248,320-token head, NFC normalizer,
  and nested EOS 248044 versus the actual runtime stop token 248046. Current
  SDK correctly rejects this binding. First audit's wrong nested-EOS assumption
  is retained; corrected audit follows the actual upstream loader, with no rerun.
- Next implement and independently audit the wider-vocabulary/native MLX binding,
  including exact bytes, Unicode literal replay and explicit stopping policy,
  before paired watermark evaluation. Do not bypass current guards or transfer
  original Qwen3-8B detector/quality acceptance to a new tokenizer.
- Qwen3.5-27B required more internal disk than available; external Archive denied
  writes. The 9B choice was made before outputs were generated. No user files
  deleted or permissions changed. This is not a new perfect-baseline launch gate.
- SDK sampler/defaults, production page, CI and public launch remain unchanged.

## September 28 longer factual writing

- Completed 32 actual unchanged-SDK outputs across four longer tasks, four
  languages and every existing key. All reach EOS; all 10,047 native tokens and
  32 distinct draw transcripts reconcile. SDK source, installed package and
  prior wheel match all 90 files; frozen engine integrity passes all 39 files.
- Matching-key diagnostics rise to 11/16 marked versus 0/16 ordinary. No
  next-slot-key hits occur. Zero outputs pass strict content, language, format
  and detection jointly. Do not trade fidelity for longer-text signal.
- Strict content passes are 0/16 in both arms; accepting all ten conservative
  judgments frozen before unblinding gives 5/16 in each. Added promises,
  invented maintenance work and wrong comparisons remain. One marked Spanish
  response inserts the English word "neither"; no wholesale translation occurs.
- Requested length/paragraph format passes are 3/16 ordinary and 2/16 marked.
  Neither reaches the generation cap. Format failures stay separate from facts.
  Joint marked passes remain zero even with all conservative judgments accepted.
- [All samples, methods, sensitivity and offline review](evidence/long-fidelity-2026-09-28/README.md).
  27 relevant local checks pass; no thresholds, defaults or criteria changed.
- This strict expansion study produces a floor effect in both arms and cannot
  isolate watermark-caused harm. Next establish ordinary-generation adequacy
  with a stronger pinned local model before paired watermark qualification.
- Local landing copy now exposes the September 28 short-study failures and
  human-review samples and corrects PydanticAI's previously stale untested label.
  Desktop/mobile navigation and existing editorial styling checked. No deploy,
  posting, public package release or CI enablement. Launch remains held.

## September 28 source-grounded fidelity and review

- Ran 96 actual generations through the unchanged public SDK: twelve fresh tasks,
  seven languages, all four existing keys, ordinary/marked pairs. Every output
  retains language, meets length limits and reaches EOS. All 6,388 native tokens
  and 96 distinct draw transcripts reconcile.
- Assistant fact-level passes are 42/48 ordinary and 36/48 marked. Accepting all
  nine flagged conservative judgments gives 44/48 and 43/48. This sensitivity
  matters; do not call the descriptive difference proven causal watermark harm.
  Clear failures still include dispatch becoming delivery, an invented weekday,
  omitted time zones/nonapproval and unsupported contact promises.
- Only 1/48 marked short outputs hits the unchanged diagnostic rule; no ordinary
  or other-key hits. These tightly constrained short tasks qualify neither
  detector power nor error rates. Longer factual-preservation tasks need joint
  content/detection testing; no scientific clue closes.
- Added an offline, metadata-hidden review page with source facts, blank human
  ratings, local persistence and explicit JSON export. QA uses separate fixtures,
  never fake human labels. Desktop/mobile, keyboard, reload, partial export,
  uncertainty and inert markup checks pass. Production design stays unchanged.
- [Evidence, all samples and review page](evidence/fact-fidelity-2026-09-28/README.md).
  29 local checks pass; 90 package files match the prior wheel and all 39 frozen
  engine files pass integrity. Publicity remains held; SDK behavior unchanged.

## September 28 five-layer comparison

- Completed 48 new actual generations: ordinary/current/five-layer task passes
  are 12/16, 14/16 and 11/16. Longer matching-key detections are 0/4, 3/4 and 1/4
  under the respective frozen rules. Reject the five-layer candidate; it improves
  neither fidelity nor detection in this development screen.
- All 48 outputs retain language and reach EOS; all 5,211 tokens and 48 distinct
  random-draw transcripts reconcile. Every sample and failure is retained.
  Assistant ratings are metadata-hidden, not independent human acceptance.
- Accepting all four ambiguous ratings still leaves the candidate behind.
  One reference output crosses the wrong-key rule. Five-layer scoring on 500
  opened null responses yields four hits, with a 2.04% IID-only upper bound:
  neither result closes detector calibration or attribution.
- [Results, full samples and methods](evidence/prefix-five-2026-09-28/README.md).
  49 local checks pass; 39 frozen engine files remain intact. SDK/website/CI
  unchanged. No scientific clue closes; publicity remains held.
- Both simple attenuation approaches failed. Next separate model factual
  limitations from watermark effects on fresh broader tasks, evaluating fidelity,
  matching-key sensitivity and wrong-key behavior jointly. Do not keep reducing
  signal strength merely to produce more varied examples.

## September 28 application-client progress

- Closed the untested PydanticAI one-turn text/native-JSON integration check on
  pinned Qwen/MLX. PydanticAI 2.51.0 and OpenAI 3.20.0 support sync/async requests,
  exact replay, and explicit typed JSON through an optional recipe.
- Four ordinary and four marked generations reach EOS; all 216 sampled tokens
  reconcile with native bytes and journals. Typed JSON preserves the tested
  names, deadline and approval condition. Replays/rejections add no inference.
- Request-ID conflicts return 409; instructions/history/streaming reject with
  400, and implicit tool output rejects in the client profile. No unsupported
  feature is silently dropped. This does not qualify agent loops or other models.
- Fresh installed wheel: all 90 package files match source; 44 relevant tests
  pass. The first usage-property harness failure is retained separately from
  the complete corrected run. [Evidence and paired samples](evidence/pydantic-ai-2026-09-28/README.md).
- No core dependencies, sampling, website design, scientific statuses or hosted
  CI settings changed. Detector/fidelity qualification and the full launch remain
  open; this is a scoped ecosystem usability improvement.

## September 28 candidate comparison

- Tested a separately versioned half-mixture candidate through 48 actual Qwen/MLX
  generations with every existing key, ordinary/current/candidate conditions,
  French repetitions, approval emails and longer English/Spanish text.
- Rejected it: rubric passes were 13/16 ordinary, 13/16 current and 12/16 mixture;
  longer matching-key detections fell from 3/4 current to 0/4 mixture under the
  unchanged nominal rule. More varied French answers did not establish better
  factual fidelity. All 48 remained in the requested language and reached EOS.
- All 5,169 committed tokens and 48 distinct draw transcripts reconcile. Ratings
  were committed before joining condition metadata. They remain assistant review;
  detector calibration and quality noninferiority are still unestablished.
- A prior SDK-facade attempt rejected the unknown research identity. That aborted
  run, errors and metadata correction are retained. The corrected caller is
  explicitly research-only; SDK identity checks remain intact.
- [Results and all long-form review samples](evidence/half-mixture-2026-09-28/README.md).
  52 relevant tests pass; all 39 frozen engine files pass integrity checks.
  No SDK default, public release, production design or CI setting changed.
- Keep the reference intact. Next work must address detection and fidelity
  together; blanket dilution is ruled out by this development screen. No scientific
  clue closes from a more varied output or a nominal, uncalibrated threshold.

## September 28 concentration diagnosis

- Eight fresh actual generations across all four existing keys isolate the
  concentration of the current 30-layer transform. Every one of 568 committed
  steps matches the separate probability reference bit for bit.
- On marked paths, 92/100 positions whose base maximum was at most 99% ended
  above 99%. Two marked answers repeat previous text exactly; two vary. The
  previously failing key still gives the incorrect upward-reflection explanation.
- The immediate next action is a separately identified concentration-limiting
  candidate evaluated jointly for task fidelity, diversity and detection power
  at the same false-positive target. Do not weaken the released profile, choose
  favorable keys or transfer detector acceptance to an altered sampling law.
- [Full measurements and limits](evidence/concentration-audit-2026-09-28/README.md).
  Diagnostic and sampling tests: 37 passed; frozen engine manifest: 39 files pass.
  No SDK default or scientific acceptance changed; publicity remains held.

## September 24 progress

- Four-key actual inference: all 72 outputs retained; ordinary 31/36 and marked
  32/36 pass unchanged narrow task rubrics. Language retained in 72/72. Marked
  French responses repeat within each key; one key repeatedly fails the rubric,
  three pass. No retries or favorable-key selection. This is evidence about key
  variation and repetition, not closure of general quality or the 25-clue ledger.
- Added an explicit wording comparison to the shared live/exported viewer.
  It preserves exact text, handles identical outputs honestly, and leaves the
  selected visual design intact. Highlighted differences are not watermarked-word
  attribution. No model calls, sampler changes or dependencies are added.
- Verified the changed viewer from a fresh installed wheel: 90 package files
  match, two real live pairs retain exact traces, custom prompts and refresh
  recovery work, and the identical-output case shows zero highlights. Desktop,
  390px mobile and keyboard checks pass. Local checks: 65 Python, 28 JavaScript.
  See [installed viewer evidence](evidence/wording-demo-2026-09-24/README.md).
- Deferred the landing page's 144-pair payload until its section approaches the
  viewport. Initial JS fell from 1,004.21 kB to 313.95 kB. All records remain;
  direct links, task filtering and recovery from a missing asset pass in browser.
  Production build and four site checks pass. No deployment or page-speed claim.
- Prepared concise [launch copy](LAUNCH_DRAFT.md) with explicit local-model and
  diagnostic scope. It remains unscheduled; no public package or repo exposure.
- Publicity remains held. Next qualification should explain and quantify
  fixed-key diversity, retain failures, and finish the scoped install/demo and
  serving acceptance work. Broader native-runtime claims stay bounded by their
  actual evidence, not protocol compatibility alone.

## September 23 quality-gate clarification

The failed gallery email is a failed sample with unresolved attribution, not a
blanket SDK launch blocker. The earlier 144-pair benchmark recorded 108 marked
passes versus 106 ordinary passes; its conservative 7.50% upper harm bound missed
the registered 5% criterion. Preserve that scientific limitation without calling
it an observed aggregate quality decline. A02 remains open; a clearly disclosed
preview does not require pretending it is closed. SDK, demonstration and launch
readiness remain separate work, and publicity is still held for that product bar.

New fixed prose comparison is retained in
[September 23 evidence](evidence/prose-quality-2026-09-23/README.md): 24 actual
generations, all in the requested language, ordinary 10/12 and marked 8/12 under
the frozen task rubric. Four repeated marked French answers missed the stricter
explanation criterion despite fresh draws. Investigate this across keys; do not
present this small pilot as quality acceptance or a closed research gap.

## Pretext productization update, September 22

- Added a reusable gallery export and prefilled explanation/email/French runs in
  the shared viewer. Switching examples updates real text, trace, measurement
  and recipe together. No dependencies or extra generation calls are added by
  export. This is a working example gallery, not completion of the creative-demo bar.
- New actual email generation violates its explicit no-time/location prompt:
  the marked result invents "the park around noon". The exact sample remains
  visible with a review note; A02 is still open. Two new French outputs remain
  French. Neither observation is a broad quality acceptance.
- The fresh installed wheel also passed live browser first-use, custom-prompt
  generation, text-edit inspection and refresh recovery on cached Qwen/MLX.
  Four outputs reconcile exactly with committed IDs. This closes this local
  installed-playground check, not Windows/Linux or first-download qualification.

- Implemented one shared comparison primitive for the live playground and Python
  (`wm.compare`, `.ordinary`, `.marked`, `.to_dict`, `.export`). No added core dependencies.
- Added a `[playground]` extra matching the CLI's platform default. Fresh Python
  3.13 environment installed the wheel and passed local model preflight.
- Exported a real Qwen/MLX English pair into the same viewer used by the live SDK.
  Static replay is labeled recorded, never runs inference, and creates matching
  Python recipes for user prompts. A second French pair ran from the installed wheel.
- Added three small remix recipes. These are starter code, not three finished
  showcase experiences. Shared viewer is about 67 KB before compression, excluding
  optional runtimes and model weights.
- Local landing preview connects the existing visual identity to that replay and
  recovers from teaching text without supplied alternatives. No production deploy.
- Added exact committed-token traces and an interactive token explorer to the
  shared live/exported viewer. Actual Qwen/MLX outputs reconstruct exactly from
  their traces. Token selection, replay, Unicode byte boundaries and reduced
  motion are covered locally. This shows selected tokens, not alternative
  probabilities or causal watermark attribution; those remain outside the trace.
- Remaining product bar: a polished creative demo gallery, richer sampling-choice
  visualization, public package installation after release gates, and broader
  scoped model/runtime and quality qualification. Existing research acceptances
  are unchanged; short diagnostic fractions do not establish detection.


Deliver a focused, dependable watermarking library with Pretext-level ease of
use and an interactive demonstration that makes its real capability visible.
Publicity remains postponed until the release candidate meets this bar.
The user's September 19 direction makes production quality of both SDK and
demos the release objective. A green test suite alone does not satisfy it.

September 20 preservation safeguard: all SDK rewrite entry points now reject
before inference with `RewriteUnavailableError`. The playground disables the
mode and rejects direct rewrite requests without consuming a run or replacing
prior results. This prevents the known unsafe rewriting path from delivering
translated or semantically altered candidates. It does not validate native model
quality, close A02, or make launch ready. The older rewrite results and UI history
below are retained as evidence; they describe behavior before this restriction.

September 20 local model expansion: the portable adapter now accepts verified
ordinary ASCII added tokens and explicitly requests non-thinking chat templates.
Pinned SmolLM3-3B completed 16 actual requests, including all six ordinary/marked
pairs, both local rewrite helpers and both local HTTP client protocols. A fresh
wheel passed 523 Python and seven JavaScript tests; 544 generated tokens matched
private journals. The larger model retains useful negation/rewrite examples,
but JSON formatting, invented email detail and a changed French deadline remain
visible failures. This advances model coverage without closing output quality,
detection or release gates. Hosted CI remains disabled at the user's request.

The optional Transformers JSON path now applies a bounded LLGuidance grammar
before base filtering and watermark sampling, then independently validates
completed output. The Python/CLI APIs and local OpenAI/Anthropic typed-client
helpers use the same path. Real SmolLM3 tests retain two fenced-JSON controls,
six successful constrained outputs, two typed-client responses and one explicitly
incomplete token cap. This addresses a format failure through a requested schema,
without repairing the older unconstrained samples or claiming semantic/detection
acceptance. The MLX path now shares this grammar contract: default reference
and experimental-fast execution each pass eleven real Qwen requests, including
both typed clients and an explicit token cap. Across those runs, 426 generated
tokens and 350 grammar masks reconcile with retained journals. The tested
Unicode values and backup prerequisite are preserved. Native framework
structured output, broader semantics and constrained-output detection remain
open; this integration does not close those gates. CI remains disabled.

## September 21 product acceptance and next steps

Vincent accepts the measured 8.40% complete-path per-token overhead for the
recorded local Qwen/MLX setup. This is scoped product acceptance, not a changed
benchmark verdict or measured throughput for all adapters. See [performance](PERFORMANCE.md).

The [compatibility matrix](COMPATIBILITY.md) now separates native runtimes,
client protocols and application SDKs. LangChain sync/async actual inference
and exact retry replay pass with Llama 3.2 3B CPU. Six retained outputs include
unrequested elaboration; the integration check does not accept output quality.
No sampling code changed and CI stays disabled.

The Pretext-level release remains unfinished. The concrete work is:

1. Complete owner review of the prepared ordinary/marked prose samples and
   define the accepted model/workload. Do not replace that with lexical checks.
2. Qualify the advertised native SGLang/vLLM serving paths: real client traffic,
   cancellation, request reuse, concurrency and failure recovery. Current CPU
   hook pilots do not meet that bar. Ollama Python protocol calls are tested;
   a native Ollama daemon sampler hook remains unavailable.
3. Calibrate any advertised detector for the actual released profile; current
   interactive diagnostic curves must not imply detection confidence.
4. Verify the final installable artifact and fresh-machine setup, then the
   prefilled real-inference demo, custom prompt, edits, errors and recovery.
   Preserve the selected website design and distinguish its teaching model.
5. Publish the clean package/repository only after those checks; verify actual
   public install and links before rescheduling publicity. No new release or
   social dispatch is implied by this update.

## September 21 routing and serving progress

Closed bounded integration gaps: native vLLM OpenAI/Anthropic HTTP, four-request
concurrency, one disconnect/recovery scenario and graceful exit; Ollama Python
sync/async protocol calls through Keyprint; LiteLLM explicit-route isolation and
timeout recovery without duplicate generation. The [matrix](COMPATIBILITY.md)
links exact versions, all outputs, failures and independent audits.

The core dependency list is unchanged. Ollama protocol handling shares the same
single model worker and error/replay/cancellation path. It does not add a router,
start another runtime or post-process generated text.

Still open: SGLang lifecycle on a restored pinned runtime, native Ollama/LM Studio
hooks, GPU and sustained serving qualification, safe automatic failover, output
quality acceptance, detector calibration and final release/demo verification.
The 22/25 historical research tally is unchanged. Publicity remains held.

## Current release gates

The [output quality contract](OUTPUT_QUALITY.md) makes language, facts, actors,
conditions and dates explicit release obligations. The actual-inference report
now marks each output blocked or unreviewed in both JSON and HTML; mechanical
passes cannot become delivery approval. This is a reporting safeguard, not a
new semantic validator or a closed research gap.

The fixed multilingual decision screen passes 128/128 exact answers (64 ordinary,
64 marked) on pinned Qwen3-8B/MLX reference execution. Sixteen positive/negative
prompts run four times per condition under a JSON grammar permitting both
answers. Receipt audits reconcile 1,047 tokens and grammar masks; fourteen
validator tests pass. [Decision quality evidence](DECISION_QUALITY.md) retains
the frozen plan, all outputs and scope limits. This is a constrained decision
result, not prose fidelity, calibrated detection or general quality acceptance.

Input-limit handling now reports tokenized input, the existing backend limit,
and the response budget through a public `InputLimitError`. Local Chat
Completions, Messages and playground requests return actionable HTTP 400 errors;
arbitrary backend error details remain private. Rejections preserve prior demo
results and replay without sampling. Actual SmolLM2 GGUF CPU checks exercise both
client libraries, generation/rewrite overflow, and recovery with six generated
outputs. The prior full suite passed 1,431 tests; the updated installed wheel
passes 213 affected tests, including 18 new limit/replay/recovery cases. This
closes an input-error usability gap, without changing sampling or token limits.

The exact earlier example-based rewrite instruction also completed a separate
28-output screen across Qwen3-8B and Llama 3.2 3B: seven sources, ordinary and
marked outputs, five previously exposed sources and two new sources per model.
There were no runtime failures. Qwen had two lexical failures from an unchanged
approval-owner pair; Llama had eleven, with translations, omissions and a Chinese
approval-condition reversal in the retained outputs. Lexical passes are not
semantic acceptance. Under the frozen selection rule, the instruction was not
promoted and the default remains unchanged. Quality and launch gates stay open.

The local playground now exposes the real SDK rewrite path. Users can switch
between separate prompt/source drafts, enter exact phrases, compare the unchanged
source with one generated candidate, inspect explicit failed checks, stop an
attempt and restore the same result after refresh. The unchanged source is never
labeled an ordinary model sample. Python rewrite helpers accept the same
cooperative cancellation Event as generation, and private receipts fingerprint
the exact rewrite prompt. An installed wheel passes 1,431 Python checks without
skips; fourteen UI checks pass. Actual Llama browser tests exercise desktop/mobile
reading, protected phrases, edit inspection, refresh without regeneration,
cancellation and worker reuse. A too-small mode selector was enlarged after
visual inspection. Numerical sampling profiles and research thresholds are unchanged.

This does not fix model quality: original-prompt Qwen rewrites copied both tested
English and French sources unchanged. A four-output prompt-development screen
produced some changed, faithful text but also changed a French weekday. A modified
example-based candidate then completed twenty confirmation outputs on Qwen and
Llama: seven Qwen and eight Llama outputs failed lexical screens, with copying,
unwanted translation and approval-actor drift retained. That candidate was
rejected and the original rewrite prompt restored. These are prompt-development
results, not an exhaustive comparison or proof against other prompt strategies.
The new UI makes failures actionable; automatic faithful rewriting and the
broader production/launch goal remain open.

The subsequent Llama 3.2 3B Q8_0 extension exposed and fixed a byte-rendering
bug: native full-sequence decoding removes French punctuation spaces. Raw
native-piece verification now preserves sampled bytes. Exact replay retains
595 original Llama tokens and 1,002 preceding SmolLM2 tokens with unchanged
sampling events. Fresh Llama runs complete 20 outputs and 2,090 tokens, plus
local client/rewrite probes. Longer English, Spanish and French cases retain
approval and backup conditions in many outputs, but invented details, factual
errors and a changed approval condition remain. Three long-form outputs fail
mechanical screens; zero flags on the six short cases still miss semantic
errors. [All pair findings and reproduction](QUALITY_REVIEW.md) are explicit.
The updated installed wheel passes 1,414 Python tests without skips; another
five harness-validation cases pass in the subsequent 49-test focused run.
No UI code changed. This advances one additional model binding and fixes an
integration defect; it does not close quality, detector or release gates.

The SDK now includes an experimental CPU llama.cpp/GGUF backend through
`Keyprint.from_llama_cpp()` and the existing CLI, local client server and
interactive playground. The preceding installed wheel passed 1,403 Python tests
without skips; 11 UI checks also pass. All 85 package files match source,
wheel and both tested installations. Its pinned SmolLM2 Q8_0 run produced 12 ordinary/marked
outputs. Independent native replay reconciles all 1,002 raw heads, probability
hashes, recorded draws, committed tokens and rendered texts. Both local client
protocols pass replay, cancellation, reuse and graceful shutdown. Exact UTF-8
cutoff replay passes in ordinary and marked conditions; Transformers replay
also passes after the shared portable-loop extraction.

The explicit 145 MB pinned download, offline cached startup, prefilled example,
custom prompt and pasted-text inspection were exercised in a real browser.
Desktop/mobile checks caught and fixed an incorrect backend label. Native
resources now close on their owning worker after requests drain. Four of the
12 GGUF outputs fail mechanical quality screens and both rewrite samples fail
checks. This adds one measured framework/model binding; it does not establish
Ollama/GPU compatibility, broader model support, quality or production cost.
See [integration scope](INTEGRATIONS.md#local-gguf-with-llamacpp) and
[local reproduction](CONTRIBUTING.md#llamacpp-local-checks). CI and launch remain
held; no clue or research acceptance is promoted by this integration.

The native caller now shares an immutable raw-head snapshot with its filter,
removing duplicate SHA-256 work while retaining all journal commitments and
validation. The preceding installed wheel passed 1,351 tests without skips;
twenty new benchmark/auditor checks passed separately. Three real MLX execution modes retain
exact frozen-reference parity across 72 outputs and 2,964 tokens. Both local
client lifecycle checks and eleven structured requests pass. A matched-revision
comparison retains 96 measured requests, four warmups and 4,144 audited tokens.
Ordinary and marked candidate/preceding-caller ratios are 0.995297 and 0.994067,
with one-sided 95% upper ratios 1.003657 and 1.000761. Both include no improvement:
this does not establish a speedup or close the failed engine-relative cost gate.
The [performance record](PERFORMANCE.md#immutable-raw-head-snapshots) preserves
scope, limitations and initial test/auditor failures. Hosted CI stays disabled,
core/native publication and publicity stay held, and the full release goal remains active.

Client installation now accepts bounded OpenAI and Anthropic version ranges
instead of forcing exact versions. Fresh environments with OpenAI 1.109.1 /
Anthropic 0.83.0 and OpenAI 3.14.1 / Anthropic 1.6.0 both resolve and pass
dependency checks, then each pass 115 focused client tests with zero skips.
Both pairs pass eleven actual Qwen structured requests with typed parsing and
exact replay: 426 tokens and 350 grammar masks across the two runs. Both pairs
also pass both real cancellation/lifecycle checks. The older pair's inference
run used the preceding wheel; all 83 package files match the candidate wheel
byte for byte. The metadata change leaves numerical pins, model bindings and
sampling identities unchanged. Existing upstream deprecation warnings and older
Anthropic parsed-object serialization warnings are retained. These two tested
pairs do not qualify every release within the allowed ranges. Hosted CI remains
disabled, and public launch remains held.

The latest candidate removes redundant MLX support-mask/index allocations while
retaining validation, immutable snapshots and journal events. Its isolated wheel
installation passes 1,335 tests with zero skips; all 83 package files match source
and wheel. Bounded reference, experimental-fast and experimental-native each pass
twelve actual Qwen pairs against the archived implementation: 72 outputs and
2,964 tokens overall. Both local client lifecycle checks and eleven structured
requests pass. A full-path audit reconciles 72 measured outputs, three warmups and
3,000 tokens. Marked/engine time per token is 1.084027 (one-sided 95% upper
1.091838), worse than the prior separate 1.072251 observation; the 1.05 cost gate
still fails. This is not a demonstrated full-path speedup. The archived reference
and native binary remain unchanged, while all current MLX caller identities bind
the new source. This is macOS ARM64 evidence using existing pinned dependencies;
no new platform, quality, detection or public-release acceptance follows.
See [performance scope and retained failures](PERFORMANCE.md#support-mask-reuse).

The installed playground now puts the real responses ahead of secondary display
controls, keeps custom prompts in a keyboard-accessible disclosure, and provides
an ordinary/marked response switch on narrow screens. Actual Qwen browser QA
passes eight interaction checks: startup, prefilled and custom generation,
exact-text switching, edit inspection, keyboard scrubbing, refresh restoration
and cancellation with visible focus recovery. Four completed EOS outputs contain
212 tokens; this is a flow check, not quality or detection acceptance. Response
panels begin at 553 px on a 1440-by-1000 desktop and 652 px on a 390-by-844 mobile
viewport, versus 886 and 1,128 px previously. Both have no horizontal overflow or
page errors. The new installed wheel passes 22 focused Python playground checks
and ten JavaScript checks; the preceding full 1,225-test result remains separately
recorded. No inference code or sampling identity changed. Hosted CI stays disabled.

The README now leads with one install command and `keyprint playground
--download` for a real prefilled/custom-text demo. This explicit flag fetches
only allowlisted files from pinned public revisions; the default remains local
cache loading. Invalid execution, ports, model/download combinations, key files
and missing native dependencies fail before a fetch. Clean MLX/server and
Transformers/server environments each pass actual CLI startup, authenticated
generation, terminal replay and session restoration: eight EOS outputs, 667
journal-matching tokens. The MLX install needs neither PyTorch nor the optional
native wheel. SmolLM2 populates a fresh cache; Qwen reuses its existing cache.
Both restart with offline library flags and no download flag. The fresh full
wheel environment passes 1,225 Python tests without skips and seven UI tests.
These first-use checks do not close semantic, detection, performance or broad
platform gates; SmolLM2 still invents email details. No launch or hosted CI run.

The CLI now offers `doctor --playground` and explicit MLX `--execution`
selection across generation, serving and the playground. Default execution
stays reference. A fresh installed wheel passes 1,186 Python checks without
skips and seven UI checks; all 83 package files match source, wheel, source
archive and installation. Actual CLI startup and authenticated playground
requests pass for native Qwen and Transformers SmolLM2: eight outputs and 487
tokens across prefilled and custom prompts, including terminal replay and
session recovery. Earlier probe-harness failures remain saved; an offline audit
reconciles all twelve generated outputs and 708 tokens. The native sampling
identity is unchanged, so the preceding cost failure below remains applicable.
SmolLM2 still invents an email date or omits the requested Friday reminder;
short caps truncate some responses. These are usability and integration checks,
not semantic, detection or release acceptance. Hosted CI remains disabled.

The native filter's full-gap range check preserves reference output bytes,
diagnostics and caller underflow policy. That preceding installed wheel passes
1,153 tests without skips; 83 package files match source. Caller parity matches
988 tokens across 24 outputs, both client lifecycle checks pass, and eleven
structured requests reconcile 233 tokens and 190 grammar masks. The unchanged
72-output engine study audits 3,041 tokens and measures marked SDK/engine time
per token at 1.108214 (upper 1.118923), worse than the preceding 1.093868 result.
Even the engine arm slowed in this separate study; background applications were
recorded, but a cause is not established. Both total-cost and within-SDK 1.05
screens fail. A separate probe confirms use of the shortcut on all 497 observed
steps. Helper speedups do not justify an end-to-end claim. The change remains
experimental, with launch held; see [current evidence](PERFORMANCE.md#full-gap-range-check).

Experimental native v2 now uses explicitly versioned lossless vector
commitments. Default/reference hashes and probability arithmetic are unchanged.
At that preceding integration stage, a fresh wheel passed 1,079 tests without skips; 83 installed files matched source
and wheel. Actual caller parity resolves vectors before comparing reference
hashes and reconciles 988 tokens across 24 outputs. Both client lifecycle checks
pass, and eleven structured requests reconcile 213 tokens and 175 grammar masks.
The unchanged 72-output engine study audits 2,972 tokens and measures marked
SDK/engine time per token at 1.093868 (upper 1.100527). That complete-path 1.05
screen failed. See the [encoding contract](tools/VECTOR_COMMITMENTS.md) and
[latest performance evidence](PERFORMANCE.md#native-v2-lossless-vector-commitments).
The preceding prototype's failed probe and later offline audit remain retained;
no cost, quality, detection, platform or public release gate is marked closed.

The preceding bounded native-selection stage preserved the complete reference filter law and
receipt format. The matching `keyprint-native 0.1.0a2` wheel is required before
model loading; an actual older wheel fails early. That installation
passes 918 tests without skips, selector sanitizer stress, full Qwen caller
parity, both client lifecycle checks and eleven structured requests. The
unchanged 72-output study reconciles 3,018 tokens and measures marked SDK/engine
time per token at 1.137564 (upper 1.146169). This remains above the 1.05 target.
No cost, quality, detection or public release gate is marked closed.

Candidate-local unkeyed setup now avoids rebuilding token profiles and parsing
tokenizer metadata per request, while preserving file hashes, binding checks,
fresh ownership and failure accounting. That preceding installed wheel passed
822 tests; all 81 SDK files matched source. Actual reference/native caller parity,
both client cancellation/reuse flows and eleven structured requests pass.
The unchanged 72-output engine study measures marked SDK/engine time per token
at 1.150419 (upper 1.158634), with 3,065 audited tokens including warmups.
This reduces the preceding 1.281173 result, but still fails the complete-path
5% screen. Setup is amortized outside request timing; cold-start improvement is
not claimed. Quality, detection and public release remain unapproved.

The optional native HMAC accelerator has a self-contained macOS ARM64 wheel
and explicitly selected SDK execution with a binary-bound identity. Clean
installation requires no compiler or separate OpenSSL. Full-caller comparison
matches all 988 reference/native tokens across twelve pairs; both local provider
clients pass cancellation, terminal replay and worker reuse. The installed SDK
at the preceding partition stage passed 798 regression tests plus four additional
failure-contract checks; all 80 packaged SDK files matched source and installation.

Before partition integration, the frozen native runtime passed the local 5% incremental timing
screen on eight declared Dolly tasks: ratio 1.028758, one-sided 95% upper
1.032226. All 64 measured outputs and sixteen warmups are retained; an audit
reconciles 10,634 tokens, original prompts, journals, text and byte rendering.
Thirty-eight measured outputs reached their token cap. Background macOS
indexing was active and recorded before measurement. This is a scoped gain over
the preceding failed SDK timing screen, not total native-server overhead or A18
acceptance. Production serving, quality, indistinguishability, calibrated
detection, platform coverage and release gates remain open. CI stays disabled.

A subsequent unmodified MLX-LM comparison exposes the remaining SDK cost:
ordinary execution takes 1.485376 times the engine's time per token, and marked
execution 1.526571 (upper bound 1.537920). All 72 measured outputs and three
warmups reconcile, including 3,053 generated tokens. Sampling and bookkeeping
differ, so this is complete-path evidence rather than a pure marking estimate.
Profiling directed optimization at full-vocabulary filtering. The partition
selector is now integrated into separately identified experimental-native
execution after exact full-filter/caller parity and both provider lifecycle
checks. A new unchanged 72-output study reduces marked SDK/engine time per token
to 1.281173 (upper 1.289455), with 3,058 audited tokens including warmups. This
is progress, but the complete-path 5% target remains unmet. Journals and numerical
contracts are preserved; further optimization and qualification remain required.

The CLI now uses the same host-aware backend default and pinned-cache fallback
for generation, serving and the playground. Explicit overrides remain available;
missing-cache errors name the pinned download command and perform no download.
Invalid prompt lengths, token caps and malformed schema JSON fail before weight
allocation. A fresh wheel passes 669 Python and seven JavaScript tests, with all
78 packaged files matching source and installation. A separate core-only install
passes demo, doctor and verify; its routing suite passes 14 checks and skips eight
optional server checks. Actual installed-CLI Qwen and SmolLM2 ordinary/marked
runs complete four outputs and 139 tokens, with exact text/token decoding and
private receipt reconciliation. Both SmolLM2 explanations are factually wrong
and remain retained. This qualifies first-use routing on Apple Silicon macOS,
not cross-platform inference, broad answer quality or a release gate. CI stays off.

Rewrites now accept explicit source phrases to preserve, validate those
settings before inference, and report missing/modified/duplicated literals.
Empty completed candidates also fail the checks. Twelve actual Qwen requests
across four languages and both provider-object helpers retain 772 tokens and
two correctly flagged verbatim-preservation failures. Three ordinary/marked
pairs are identical despite independent draws and changed prepared weights.
The fresh wheel passes 647 Python tests; all 78 packaged files match the tested
installation and source. This gives callers more precise fidelity controls and
original/candidate comparisons, but does not guarantee literal compliance,
semantic fidelity, quality preservation or detection. Those gates remain open.

Native completion handling now verifies host text against returned token bytes,
rejects unknown finish reasons, and retains incomplete UTF-8 tails at exact
token limits. An actual SGLang stress run exposed a separate lifecycle bug:
completed request cycles exhausted the 32-live-session bound. That failed run
retains 32 outputs and exit code 137. Completion-state cleanup fixes the leak
without raising the active-request limit or permitting finished requests to
restart. The full rerun passes 72 requests, 747 journal-matching tokens and 27
partial-character outputs across six multilingual/emoji prompts and six caps.
Five native contract checks pass; the fresh wheel passes 626 Python tests and
all 78 packaged files match the installed copy and tested source. Reconciliation
checks short token paths as a complete multiset, not unique request identities.
This qualifies this bounded SGLang CPU test, not vLLM cap behavior, streaming,
long-running service, answer quality or detection. Public launch stays held.
A separate standard comparison on this source reconciles twelve outputs and
949 tokens, including eleven EOS completions and one cap. Five mechanical
quality failures remain visible; this does not close the quality gate.

The playground now serves its page while the model loads, exposes real startup
state, permits prompt editing and blocks premature generation. Failed startup
keeps the page available, with private diagnostics and explicit reconnection.
Installed-wheel QA passes 582 Python and seven JavaScript tests. Desktop/mobile
Chrome checks retain one actual Qwen pair (91 generated tokens) plus an edited
inspection; refreshing starts no new generation. An explicit QA loading barrier
and injected loader failure test the waiting/error states without claiming a
startup-speed measurement. This improves first-use behavior but does not close
the full first-use, demo or release gates.

The September 20 prompt-free surrogate-likelihood development candidate failed
its frozen power gate: 7/12 marked answers detected, including 0/5 short answers,
with zero ordinary or wrong-key flags in this 24-text development sample. Its
log-space calculation avoids probability flooring; the preceding numerical
attempt remains retained as 24 scoring errors, not negatives. All 16,135 retained
model-head hashes, text/token identities and fixed prefixes match the earlier
fixed-batch inference receipts. This identity check is not a score or calibration
audit. No threshold adjustment, larger null run or SDK promotion follows this
failed screen. Short-text detection and public launch remain open.

A subsequent model-centered bit betting mixture also fails its frozen gate:
8/12 marked detections, only 1/5 short detections, and zero paired negative-check
flags. All 24 opened responses are retained. Ten mathematical/contract tests and
a bounded diagnostic audit pass, but neither is detection acceptance. This
candidate remains outside the SDK; no larger null run or cutoff change follows.
`INFERENCE_TESTING.md` compares the tested formulas and their limitations.

Fixed 10-to-1 relative layer weighting of those same model-centered residuals
improves opened-data results to 9/12 overall and 2/5 short detections, still below
the unchanged 10/12 and 4/5 gate. All 24 responses are retained; 48 aggregations
and 32,270 terms independently reconcile. A separate diagnosis on the exact
same 1,496 positions in all five short marked responses finds original-prompt
generation likelihood above the reference cutoff in 5/5 cases, versus 0/5 for
the prompt-free surrogate. This localizes a conditioning-information problem,
not a solved detector: original prompts/probabilities are private oracle inputs,
and no fresh negative controls were run. Detection, SDK promotion and launch
remain open. Both calculations reuse archived evidence; hosted CI stays disabled.

Actual original-prompt replay now reconstructs probabilities without private
generation logs or saved probabilities. On all 24 opened responses, the unchanged
half-mixture score and cutoff detect 12/12 marked answers, including 5/5 short
answers, with zero flags in 36 paired negative checks. An independent full-term
audit reconciles 48 scores, 32,270 terms and 29,812 tournament transforms within
3e-14 score units; 151 shared-prefix model heads match. Thirty-six focused checks
pass. The development gate passes for this prompt-aware input contract. Fresh
tasks, new keys, empirical false-positive validation and prompt-free detection
remain open. SDK behavior and launch gates are unchanged; this research result
does not turn the interactive bit fraction into a detection verdict.

Fresh prompt-aware confirmation now passes on twelve previously unused
exact-field-disjoint task groups and two new keys: 12/12 marked hits, including
11/11 answers in the 100–400-word range, with zero flags across 36 paired negative
checks. All 24 outputs and twelve token caps remain retained. An independent
audit reconciles 9,957 generated-token records, 48 scores and 19,886 terms within
2e-14 score units. Fifty-three focused tests pass; the full comparison report
also passes desktop/mobile and exact-text browser checks. This establishes a
scoped fresh sensitivity result, not a population false-positive bound or a
prompt-free detector. Larger null validation remains necessary. Source-grounded quality
issues, broader model/framework qualification and serving-cost gates remain open.

A frozen local false-positive study attempted 500 previously unused
exact-field-connected Dolly groups, using the same two confirmation keys and
unchanged prompt-aware score. The disk guard stopped model work after 288 scored
documents, with zero flags. The remaining 212 are retained as unavailable. The
study is incomplete and fails its all-500-available gate; no final false-positive
bound is reported. An independent audit reconciles all 500 attempt records,
60,054 model heads, 576 scores and 120,108 terms. Integrity passes; the statistical
screen does not. The original attempt must remain unchanged through any separately
declared operational recovery. Hosted CI remains disabled.

A separately declared recovery completed all 212 previously unmeasured
disk-guard cases, retaining the 288 valid original cases without regeneration.
A cache-disabled preflight on three already scored controls reproduces all 588
raw heads and six scores exactly, with about 5.6 GB tracked peak memory. Source,
keys, model and scoring rules remain fixed; compact receipts reduce storage.
All 500 controls are available with zero flags or errors. The numerical audit
reconciles 106,338 heads, 1,000 scores and 212,676 terms, with maximum score
discrepancy 2.85e-14; the separate provenance audit also passes. The one-sided
97.5% IID-only upper bound is 0.7351%, conditional on independence this corpus
cannot establish. The original no-retry attempt stays incomplete; the recovered
sample cannot be described as fresh. Thirty-eight focused checks pass. SDK
behavior and public release gates are unchanged.

An offline evidence notebook now combines all twelve generated pairs and all
500 control attempt records. Exact text, category filters, keyboard record
navigation, mobile response switching and explicit unavailable states pass
rendered Chrome checks on the original incomplete study. Eight export-integrity
tests plus thirty null/recovery tests pass. The final recovered result also
passes rendered Chrome desktop/mobile checks, including both execution origins,
the original failure and conditional IID-only bound. SDK behavior and release
gates remain unchanged.

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
  or an equivalent local release matrix and clean-install checks pass;
  publication and announcement claims match evidence. Hosted CI stays disabled
  under the user's current instruction.

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

Default MLX generation now uses the unchanged frozen V3 sampling host with a
separately identified bounded UTF-8 finalizer. The archived strict research
caller is preserved. A fresh installed wheel passes 508 Python tests; twelve
real Qwen ordinary/marked pairs compare the new caller with independent frozen
executions, matching all 494 tokens per path, probability/draw records, text,
literal diagnostics and consumed work. Independent reconciliation verifies all
988 commits. The original 192-token failure also replays successfully through
the new default with all 192 model heads and 302 draws matched and pending bytes
e2 9c retained; the original failed study remains unchanged. Real OpenAI and
Anthropic clients pass text/usage receipt and replay checks. Actual
Anthropic-client cancellation retains consumed work, replays the terminal error
and permits a fresh response on the same worker. Complete output, EOS and manual
pipeline finalization retain their strict behavior. These are scoped correctness
checks, not quality, calibration or serving-cost acceptance. Native framework
cap handling remains separate work.

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
