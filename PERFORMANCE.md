# Sampling performance

Anthropic states that watermarking has negligible speed impact and adds no
extra watermark tokens or serving price. It publishes no numeric latency
budget. The 5% upper-bound screen below is our engineering operationalization,
not an Anthropic requirement.

In the last completed three-arm support-mask study, the marking increment
within Keyprint passed that screen: 3.48% observed, 3.90% one-sided upper bound.
The complete Keyprint path versus native MLX-LM failed: 8.40% observed, 9.18%
upper bound. The latter includes different sampling machinery and durable
journals; it does not isolate watermark-transform cost. See the exact study
limits below. The subsequent snapshot revision has no fresh native-engine
baseline. Neither denominator establishes production batch throughput.

**Product acceptance, September 21:** Vincent accepts the observed 8.40% total
per-token overhead as negligible for that measured local setup. This removes
that setup's product latency objection. It does not change the frozen 5% screen,
its failed result, the historical A18 research status, or qualify unmeasured
frameworks, current revisions, HTTP overhead or production batch throughput.
The one-sided upper bound remains 9.18%.

Source: [Anthropic announcement](https://www.anthropic.com/news/claude-text-watermark).

The latest complete-path comparison fails the lightweight-runtime target:
marked Keyprint takes 1.0840 times MLX-LM's time per token on the fixed local
workload. The much smaller marking-only cost measures a different denominator.
The latest one-sided 95% upper ratio is 1.0918. The preceding separate study
measured 1.0723; independent outputs and background activity prevent
attributing those differences solely to code. The unchanged 1.05 screen fails.
See [support-mask reuse](#support-mask-reuse) before using
any of the helper or within-SDK results below as a performance claim.
The newer [raw-head snapshot revision](#immutable-raw-head-snapshots) has a
matched-revision comparison only. Its small point-estimate improvement is not
statistically established and does not supersede that failed engine-relative
screen or qualify current total overhead.

The [lossless vector encoding](tools/VECTOR_COMMITMENTS.md) is integrated only
into explicitly selected experimental native v2 execution. Default and frozen
reference receipt formats keep their existing dense hashes.

The portable Transformers backend and experimental SGLang/vLLM adapters use
sparse execution of the existing binary64 sampling law. Post-filter excluded
logits and zero-probability intervals no longer cross Python arithmetic loops.
The reference `math.exp`/`math.fsum` operations, exact integer ratios, rejection
sampling and original token indices remain unchanged. Reports identify the
execution source independently. The frozen MLX/reference engine is unchanged.

## Reproduce the arithmetic comparison

From an installed checkout:

```sh
python tools/benchmark_sampling.py --output sampling.json
```

The benchmark uses synthetic post-filter logits, a fixed fixture key, marked
weights, and identical random seeds. It alternates execution order and verifies
bitwise probabilities, selected vocabulary indices and complete draw records.
It excludes model inference, support filtering, the watermark transform,
journaling, loading and text inspection. It is not a serving benchmark.

September 17, 2026, local macOS ARM64, Python 3.13.13, NumPy 2.5.2, 20 iterations:

| Vocabulary / positive candidates | Dense median | Sparse median | Ratio |
| --- | ---: | ---: | ---: |
| 49,152 / 100 | 22.56 ms | 0.210 ms | 108× |
| 151,669 / 100 | 70.74 ms | 0.394 ms | 180× |

These ratios apply only to that arithmetic microbenchmark. Measurements vary
with hardware, support size and system load; they are not a watermark-overhead
comparison against an ordinary production inference framework.

## Real-model parity check

A local CPU float32 SmolLM2-135M-Instruct run compared the old dense operations
with sparse execution, using the same fixture key, prompt and random-bit stream.
Both conditions completed at EOS: 29 ordinary tokens and 58 marked tokens.
Token paths, rendered text, prepared-weight hashes and every random draw matched
exactly. Observed marked generation time was 2.54 s versus 1.25 s. This single
pair is a functional parity check; execution order, concurrent work and cache
warmth were not controlled sufficiently for a throughput claim.

Tests additionally cover tied logits, the supported logit-gap boundary,
subnormal weights, one-candidate support, rejected draws, invalid inputs and
support-loss failures before randomness is consumed. Exact arithmetic parity
does not establish semantic quality, calibrated detection or production readiness.

## Remaining work

### Paired local serving experiment

```sh
python tools/benchmark_serving.py --model models/qwen3-8b-4bit \
  --output private-serving-benchmark --idle-host-confirmed
```

Run only after other inference studies have finished. The explicit flag records
the operator's check, not proof that the operating system was completely idle.
The output filesystem must have at least 2 GiB free before receipt creation or
model loading. This preflight prevents known low-space starts; it cannot reserve
space against other processes.
The command freezes its declaration before loading the model, measures model
loading separately, retains two declared warmups, then runs four ordinary/marked
pairs for each of six fixed prompts. Execution order alternates. Every response,
EOS, cap and failure stays in the receipts; failed or missing pairs prevent a
completed timing summary.

The primary measure is the geometric mean marked/ordinary seconds per committed
token. A fixed-seed paired bootstrap resamples within each prompt. Its one-sided
95% upper ratio must be at most 1.05 to pass this **incremental timing screen**.
Request latency is also reported because output lengths can differ. The four
pairs per prompt provide only approximate fixed-workload timing uncertainty;
they do not represent arbitrary tasks or production load.

Timing includes the SDK's model calls, filtering, marking, sampling, durable
journals and report serialization. It excludes inspection, HTTP transport and
streaming. Model-load timing excludes Python/package imports, and the filesystem
cache may already be warm. Each request records MLX active, peak and cached
memory; OS peak RSS is cumulative, not a per-request memory comparison.

The ordinary arm uses this same SDK. A timing pass would therefore not establish
total overhead against a native inference server, production batch throughput,
or acceptance of A18. No timing result is claimed merely because the harness or
its statistical checks pass tests.

### September 20 real-inference result

The pinned Qwen3-8B / MLX experiment completed all 48 measured requests and
two retained warmups without generation errors. Every measured output reached
EOS. The four pairs for each of the six fixed prompts produced 965 ordinary
tokens and 980 marked tokens. The exact texts, caps and measurements remain
in the receipts; nothing was replaced or excluded after measurement.

| Measure | Result |
| --- | ---: |
| Marked/ordinary seconds per committed token, geometric mean | 1.0763 |
| One-sided 95% bootstrap upper ratio | 1.0802 |
| Whole-request latency ratio, geometric mean | 1.0692 |
| One-sided 95% whole-request upper ratio | 1.0921 |
| Separate model load, including asset verification | 5.18 s |
| Maximum measured-request MLX peak allocation | 4.52 GiB |

The **5% incremental timing screen failed**. This is about 7.6% incremental
cost per token within the same SDK, not negligible total overhead against a
native server. The bootstrap covers this small fixed workload only. No A18 or
production-serving acceptance follows, and the prespecified limit is unchanged.

`tools/audit_serving.py --run PRIVATE_RUN --model PINNED_MODEL` independently
reconciled 2,009 committed tokens including warmups, original prompt hashes,
conditions, artifact hashes, journal chains, displayed texts and the primary
point estimate. It also reproduced the declared bootstrap summary. This checks
stored receipts and arithmetic, not fresh model-head replay or OS isolation.

The audit produces `public/comparison.html` and `public/comparison.json` with
all 24 side-by-side pairs. All eight JSON outputs parsed to the exact requested
values, and the backup instruction retained its restore-confirmation condition
in all eight outputs. All eight French outputs changed `09:30` to `09h30`,
failing the declared exact-literal screen while retaining the same time.
Science outputs also varied in their additional distance/altitude claims.
These observations do not establish semantic equivalence, reader
indistinguishability or watermark-caused quality differences.

### Profiling and a bounded PRF prototype

`tools/profile_serving.py --model PINNED_MODEL --output PRIVATE_PROFILE` retains
one real ordinary/marked pair for every fixed case, with per-request cProfile
data and texts. The September 20 run completed twelve outputs without errors:
257 ordinary and 255 marked tokens. The dense softmax path repeatedly visits
the entire 151,669-entry mapped vocabulary in Python despite top-k filtering.
Both conditions incur this cost. Marked generation additionally spends time in
per-label HMAC computation and the 30 tournament updates. Instrumentation changes
these costs substantially; its timings are diagnostic, not serving measurements.

`tools/benchmark_prf_context.py --output PRIVATE_PRF_SCREEN` tests a local HMAC
context-copy prototype without changing the SDK. It keeps the exact address
encoding and uses no global or cross-request key cache. All 540 synthetic
comparisons matched the frozen bit table. Across the measured contexts, median
candidate/reference time was 0.852–0.879 with ten labels and 0.826–0.844 with
100 labels; one label regressed to 1.152–1.302. Do not omit that regression or
turn these helper timings into end-to-end speedup claims.

The prototype also matched the frozen PRF at all 476 nonempty-label contexts
in the twelve actual inference paths, covering 14,280 bits with the run's
original key. This does not verify full generation under a substituted runtime,
other cryptographic implementations or the 5% serving limit. Tests cover binary
and Unicode labels, long prefixes, alternate keys and custom profile methods.
The prototype remains in `tools/`; no global monkeypatch or SDK promotion was
introduced. A separately identified execution path and full parity checks are
required before using either optimization in generation.

### Separately identified experimental execution

`keyprint.experimental.fast_mlx.pipeline` now provides an explicit supplied-head
pipeline combining sparse softmax execution with response-local HMAC context
copies. Frozen reference files remain unchanged. The new runtime digest binds
the experimental source and sparse arithmetic implementation; its score namespace
remains the reference namespace. No reference acceptance transfers automatically.
The low-level factory still constructs a bound candidate. For repeated generation,
`Keyprint.from_mlx(..., execution="experimental-fast")` now retains one bound
candidate and creates fresh response state. Its separate caller preserves the
reference durable journal, random-draw and commit ordering. A separate reporting
contract validates experimental source identity without weakening the frozen
reference contract. Default generation and CLI behavior remain unchanged.

```python
from keyprint import Keyprint

watermark = Keyprint.from_mlx(
    "models/qwen3-8b-4bit", key=Keyprint.new_key(), execution="experimental-fast"
)
result = watermark.generate("Explain why the sky is blue.", max_tokens=128)
print(result.text)
```

```sh
python tools/validate_fast_mlx.py --model PINNED_MODEL --output PRIVATE_PARITY_RUN
```

The September 20 freshly installed wheel completed twelve actual Qwen outputs
across all six fixed cases and both conditions. All 494 model heads produced
identical base/prepared probability hashes, sampled tokens, rejection transcripts,
rendered increments, final text and diagnostic counts in reference and optimized
pipelines. Both receive the same real model head and public-fixture random stream;
this is a lockstep execution test, not independent stochastic quality validation.
Every receipt and journal is retained and the recorded token paths were separately
reconciled. HMAC state cleared on closure of every response.

Tests also exercise repeated contexts, explicit random-draw rejection, reasoning
and tool routing, proofread source protection, invalid heads/randomness,
commit-before-control-error behavior, key changes and context isolation. The
earlier pipeline-only wheel's local suite passed 276 tests. The complete-caller
validator now runs independently executed reference and experimental model
paths with identical explicitly supplied fixture randomness:

```sh
python tools/validate_fast_caller.py --model PINNED_MODEL --output PRIVATE_CALLER_RUN
python tools/validate_cancellation.py --backend mlx --execution experimental-fast \
  --model PINNED_MODEL --output PRIVATE_CANCELLATION_RUN
```

The installed caller wheel passes 290 local tests. All twelve real Qwen pairs
(24 outputs) match across 494 committed tokens per execution path, including
sampling records, final text, literal diagnostics and consumed-work reservations.
Every output reaches EOS. The actual OpenAI-client-over-TCP check cancels after
one committed token, retains its receipt, replays without regeneration and
successfully generates 32 tokens on the same worker. A disclosed post-forward
barrier controls cancellation timing; this does not establish kernel preemption
or natural cancellation latency. Full production lifecycle, fresh end-to-end
serving measurements and exact-revision hosted checks remain required. The earlier 7.6% incremental cost and failed 5% screen describe the
unchanged default path; this parity result does not replace those measurements.

### Interleaved reference and optimized serving study

```sh
python tools/benchmark_fast_serving.py --model PINNED_MODEL \
  --output PRIVATE_INTERLEAVED_RUN --idle-host-confirmed
python tools/audit_fast_serving.py --model PINNED_MODEL \
  --run PRIVATE_INTERLEAVED_RUN
```

This separate study freezes its declaration before model loading. It retains
four 32-token warmups and 96 measured requests: six fixed cases, four repeats,
two execution paths and two conditions. Each execution/condition cell visits
every order position over the four repeats. One verified model/tokenizer is
shared sequentially, with fresh response caches and a bound candidate per path.
The study preserves the original benchmark script and its failed result.

The primary measure remains the optimized path's marked/ordinary seconds per
committed token, with the unchanged one-sided 95% bootstrap upper limit of 1.05.
Fast/reference comparisons are secondary: a faster implementation can still
fail the incremental watermark-cost screen. Independent cryptographic draws
can produce different text and output lengths. The intervals describe this
small fixed workload, not arbitrary workloads or native-server throughput.
Loading both candidates with one shared model is recorded separately; it does
not provide separate cold-start distributions for the two execution paths.
The auditor reconciles every declared attempt, prompt, condition, artifact hash,
journal chain, committed count, displayed text and timing summary. All generated
pairs remain visible, including mechanical review flags and capped outputs.

September 20 result: all 96 measured requests reached EOS, with no errors.
The audit reconciled 4,028 committed tokens including the four retained warmups.

| Measure | Geometric mean ratio | One-sided 95% upper ratio |
| --- | ---: | ---: |
| Optimized marked / optimized ordinary, seconds per token | 1.0700 | 1.0744 |
| Reference marked / reference ordinary, seconds per token | 1.0666 | 1.0725 |
| Optimized / reference, ordinary seconds per token | 0.5670 | 0.5693 |
| Optimized / reference, marked seconds per token | 0.5688 | 0.5712 |

The optimized implementation used about 43% less time per token in both
conditions on this workload, while its incremental marking cost still failed
the 5% screen. Faster overall generation does not close A18. Whole-request
marked/ordinary ratios for the optimized path were 1.0838 (point) and 1.1004
(upper); different generated lengths contribute to these values.

All 48 ordinary/marked pairs are retained. Sixteen French outputs changed the
exact required `09:30` spelling and triggered mechanical flags. The sixteen JSON
outputs matched the requested fields and values, and the sixteen backup
instructions retained the restore-confirmation condition. These small repeated
cases do not approve semantic quality or reader indistinguishability. Next,
profile the optimized marked path and preserve this result before proposing
another separately identified implementation or declared timing experiment.

### Follow-up marking-path optimization

`tools/profile_fast_serving.py` retained twelve actual optimized Qwen outputs
(247 ordinary and 249 marked tokens). Instrumented marked costs included HMAC
context copies and the 30 tournament updates. These profiler costs are not
uninstrumented serving timings.

Two separately retained development screens tested those paths. NumPy tournament
updates preserved all 4,800 layer comparisons and subnormal counters. They took
about 0.79 of reference helper time at 100 candidates and 0.48 at 1,000, but
regressed to 4.17 at ten and 11.27 at one. This candidate stays in `tools/` and is
not used by the SDK.

The SHA-context helper uses the standard SHA-256 HMAC inner/outer pad construction
for the fixed 32-byte SDK key, implemented with `hashlib`. It preserves the exact
address serialization, full digest and final one-bit extraction. It matched all
540 synthetic comparisons and used 0.618–0.691 of the existing reusable-HMAC
helper time across one, ten and 100 labels. These are helper timings, not an
end-to-end speedup claim.

The explicit experimental MLX path now uses that SHA context with fresh state
per response; closing a carrier clears its context references. Its runtime digest
binds the added `hmac_context.py` source. The frozen reference remains unchanged.
A fresh installed wheel passes 353 tests, including full-digest stdlib comparisons,
invalid keys, binary/Unicode addresses, subnormal arithmetic, routing, journal
failures and cancellation. Complete-caller parity matches 24 real outputs and
494 tokens per execution path. The separate auditor reconciles 988 token commits,
prompts, identities, journals and report hashes. Actual Qwen HTTP cancellation,
terminal replay and worker reuse also pass for this new source identity.

```sh
python tools/audit_fast_caller.py --run PRIVATE_CALLER_RUN
```

The separate repeated serving study completed all 96 measured requests at EOS
with no errors. Its audit reconciled 4,102 tokens including four warmups. The
optimized marked/ordinary seconds-per-token ratio was **1.0607**, with a one-sided
95% upper ratio of **1.0684**: the unchanged 5% screen still failed. Reference
marked/ordinary was 1.0783 (upper 1.0829). Optimized/reference ratios were 0.5672
for ordinary output and 0.5580 for marked output. These within-run comparisons
show about 43–44% lower time per token than the reference on this fixed workload.
They do not isolate the new helper's incremental effect against the previous
optimized version across separate runs.

All 48 pairs are retained. Sixteen French outputs still triggered the exact-time
format screen. Earlier timing failures remain retained; no A18, semantic-quality,
reader-indistinguishability or native-server acceptance changes. Further changes
must preserve parity and undergo a new prospectively declared serving run.

### Batched tournament candidate

The original per-round NumPy prototype remains retained. A follow-up batches
all 30 rounds, validates the immutable bit table once and keeps `math.fsum`,
operation order, per-round normalization checks and explicit subnormal flooring.
A separate fixed-seed screen matched all 8,400 layer probabilities and diagnostic
counter comparisons. At 64–1,000 candidates it used 0.480–0.505 of scalar helper
time. One candidate regressed to 2.58 and ten to 1.13; matrix preparation is
included in these timings. The experimental SDK therefore dispatches to batching
only at 64 or more eligible candidates, retaining scalar updates below that.
The threshold is an execution choice, not a detector or acceptance threshold.

The new `batched_tournament.py` source is bound into the experimental identity.
The prepare lifecycle body remains identical to the frozen reference; only its
separate module's transform selects the batched arithmetic. Protected token
probabilities, excluded support, exact sampling and context cleanup remain under
parity checks. The default reference implementation is untouched.

The installed batched wheel passes 388 tests. Complete-caller parity again
matches 24 real outputs, 494 tokens per execution path and all sampling records,
text and literal diagnostics. The independent auditor reconciles 988 commits;
actual Qwen HTTP cancellation, terminal replay and worker reuse also pass.
These checks include the 63/64 dispatch boundary, protected candidates, zero
support, subnormal roundups and caller failure accounting.

The follow-up serving run completed 96 measured requests at EOS with no errors.
The auditor reconciled 4,100 tokens including four warmups. Optimized
marked/ordinary seconds per token was **1.04354**, with one-sided 95% upper ratio
**1.0498404**. It narrowly passes the unchanged 1.05 development screen. The
whole-request ratio was 1.05177 (upper 1.07568), including differing output
lengths. Optimized/reference seconds-per-token ratios were 0.57116 for ordinary
output and 0.55399 for marked output, measured within the same interleaved run.
All 48 text pairs remain retained, including sixteen French exact-time flags.

This first development-workload pass does not close A18. The small margin, prior
sequential experiments, separate startup distributions, broader workloads,
native-server overhead and exact-revision hosted checks require follow-up.
Freeze this implementation before an independently declared confirmation run;
do not extend this consumed run or replace its preceding failures.

These are sequential development experiments on a fixed workload. Even a timing
screen pass would require independent confirmation and wider workloads before
production or negligible-overhead claims.

### Frozen-candidate confirmation workload

`tools/serving_confirmation.py` binds the exact experimental runtime digest before
collecting timings. It selects one source-hash-ranked record from each of the
eight categories in the pinned Databricks Dolly corpus, preserving its instruction
and context. Selection uses no generated output. Records must have a nonempty
instruction and a formatted prompt of at most 8,000 characters. The same source
was used in earlier research; these tasks are new to the timing workload, not a
claim that the entire corpus was previously unseen.

The declaration retains sixteen 32-token warmups, one ordinary/marked pair per
case, followed by four measured pairs per case: 64 measured requests. The output
cap is 192 tokens; every EOS, cap and failure stays in the result. The primary
screen remains the one-sided 95% upper marked/ordinary time-per-token ratio at
most 1.05, using the unchanged paired-bootstrap implementation. Whole-request
ratios and memory remain descriptive. This confirmation does not measure native
server overhead, separate cold-start distributions or deployment load.

```sh
python tools/serving_confirmation.py --model PINNED_MODEL \
  --corpus PINNED_DOLLY_JSONL --expected-runtime FROZEN_RUNTIME_SHA256 \
  --output PRIVATE_CONFIRMATION --idle-host-confirmed
python tools/audit_serving_confirmation.py --model PINNED_MODEL \
  --corpus PINNED_DOLLY_JSONL --run PRIVATE_CONFIRMATION
```

The auditor regenerates selection from the original corpus and reconciles every
prompt, condition, cap, runtime, artifact hash, journal chain, committed count,
text and timing summary. The comparison artifact attributes Databricks and retains
the corpus's CC BY-SA 3.0 notice. Timing integrity does not approve the answers.

The frozen-candidate confirmation completed all 64 measured attempts: 29 reached
EOS, 34 returned capped output, and one ordinary request failed during finalization
at its 192-token cap. The summary is **incomplete**, with null timing analysis;
the independent audit rejects it. Do not remove the failed request or transfer
the earlier narrow development pass into confirmation acceptance.

Exact replay of the failed classification request matched all 192 raw model-head
hashes and reused all 302 recorded random draws. Its final token had bytes
`20 e2 9c`; the decoder retained `e2 9c`, an incomplete UTF-8 sequence, and raised
`UnicodeDecodeError` at finalization. The replay retains a diagnostic prefix and
byte evidence separately; it does not change the original failed attempt. An
initial replay-tool cleanup error was fixed and its artifacts were retained.
This is now a concrete SDK release blocker: handle a token budget ending inside
a character without inventing replacement bytes, drawing extra tokens beyond
the cap or discarding consumed-work evidence.

`tools/render_confirmation_attempts.py` displays all 64 original attempts,
including the error and truncation flags, without a timing acceptance result.
The SDK source remained frozen throughout this confirmation. The new harness's
14 focused tests passed; this does not close the observed runtime failure.

### Token-limit finalization repair

The subsequent experimental runtime has an explicit `finish_at_limit` path.
It verifies the exact reached token budget and retains the valid UTF-8 prefix,
every committed token ID and the decoder's pending bytes. No replacement
character or additional generation step is introduced. EOS, control boundaries,
invalid bytes and ordinary supplied-head `finish()` remain strict. A capped
carrier has unavailable literal-replay diagnostics rather than a substituted
generation-path score. The frozen reference and default caller are unchanged.

The fixed-runtime replay matched all 192 original model heads and 302 random
draws, retained the original prefix and `e2 9c` suffix, and returned `length`
with unchanged consumed-work counts. The original failed confirmation is still
failed. The replay's first validation harness had a variable-name collision;
its journal and failure log were retained, the harness was corrected, and a
separate replay passed. This is a regression repair, not new timing evidence.

The newly installed wheel passes 415 tests. Coverage includes all UTF-8 prefix
lengths, strict non-cap failures, zero-budget behavior, generated channels,
reuse, and a fixture-driven OpenAI-client length response with idempotent replay.
Separately, actual Qwen inference through the OpenAI client over TCP passes
cancellation after one committed token, terminal replay and a new 32-token
response on the same worker. Hosted CI remains unavailable because the account's
Actions jobs cannot start. Fresh serving confirmation is still required.

```sh
python tools/validate_capped_utf8.py --run ORIGINAL_FAILED_CONFIRMATION \
  --attempt ORIGINAL_FAILED_ATTEMPT --model PINNED_MODEL \
  --postmortem ORIGINAL_REPLAY_POSTMORTEM --expected-runtime FIXED_RUNTIME_SHA256 \
  --output NEW_PRIVATE_REPLAY
```

### Post-repair repetition of the declared workload

The fixed runtime completed 64 measured requests plus sixteen warmups without
errors: 36 measured outputs reached EOS and 28 reached the token limit. This
repeats the same eight declared Dolly tasks with new random draws; it is not an
unseen-workload study. The independent auditor reconciled 10,727 committed tokens
including warmups and 16,441 random draws. New byte checks reconstruct rendered
text from all sampled token bytes, check pending-byte receipts and reconcile
rejection draws, commit order and consumed-work totals. All eighty outputs ended
on complete UTF-8 boundaries, so the original failure's exact replay and regression
tests remain the direct evidence for partial-character handling.

Marked/ordinary seconds per committed token were **1.053402**, with a one-sided
95% upper ratio of **1.057928**. The unchanged 5% screen **fails**. Whole-request
latency was 1.059622 (upper 1.101163). The earlier narrow development pass does
not establish negligible serving cost on this workload. All 32 measured text
pairs and 28 truncation flags remain visible. No production or A18 acceptance
was added.

A post-hoc, unblinded assistant review covers all 24 outputs from classification,
closed QA and information extraction. Four marked classification outputs and
two ordinary outputs contradict the pinned dataset's answer by classifying
Zidane as a player who never won. One further ordinary response ends before
classifying the full list. Both paths also add information absent from the
provided passages. These are source-bound observations, not a blinded quality
test or a causal estimate of watermark harm. The remaining forty measured
outputs have not received that source review; quality stays open.

The byte/draw auditor passes 21 focused tests, including serialized SDK reports,
zero-draw deterministic samples and corrupted receipt rejection. Initial audit
harness failures were retained and corrected before the final reconciliation;
SDK execution and the serving study were not changed. To profile this workload
without presenting instrumented timing as serving evidence:

```sh
python tools/profile_fast_serving.py --model PINNED_MODEL \
  --confirmation PRIVATE_CONFIRMATION --output NEW_PRIVATE_PROFILE
```

Profiling retained sixteen additional outputs on those same tasks (2,647
committed tokens, no errors). Under instrumentation, marked-path HMAC bit
production accounted for about 3.47 ms per token cumulatively; these numbers
are diagnostic, not serving timings. A development-only prototype moved all
layer-bit generation into one Python call. All 2,115,000 compared bits matched,
but its median helper-time ratios ranged from 0.9632 to 0.9877 across five sizes.
That small helper-only gain does not establish a useful end-to-end improvement.
It remains outside the SDK; no repeated serving run or acceptance was inferred
from it. Reproduce with `tools/benchmark_sha_bits.py --output NEW_PRIVATE_SCREEN`.

### Native PRF development screen

The next experiment moves a complete HMAC batch across the Python/native boundary
once. `tools/native_prf/` contains an explicit local builder and a bounded OpenSSL
3 EVP helper. It is not loaded by the SDK or built during package installation.
It returns every digest byte for auditing; reference address serialization,
the fixed 32-byte key and SHA-256 HMAC construction remain unchanged.

On macOS ARM64 with OpenSSL 3.6.3 and Apple clang 21.0.0, all 2,115,000 timed
full-digest comparisons and 10,890 SHA boundary comparisons matched the existing
Python SHAContext and stdlib HMAC. Median native/reference helper-time ratios
were 0.2340, 0.1820, 0.1898, 0.1883 and 0.1890 at 1, 10, 64, 100 and 1,000 labels.
Context construction, ctypes input packing and complete output copying are
included. These helper timings do not measure the complete SDK bit-table path,
model generation, native-framework overhead or serving acceptance.

All 60 focused native/SHA checks pass, including bounds, binary/Unicode inputs
and concurrent independent keys. A separate native address/undefined-behavior
sanitizer harness passes 10,000 repeated calls and output-canary checks. The
linked OpenSSL library was not rebuilt with sanitizers. Replaying addresses from
all sixteen retained Dolly profile outputs verifies another 77,490 complete
digests across 2,647 committed tokens, with report hashes, journal chains and
commit counts reconciled. This reuses existing texts; it is not new inference or
full model/sampling replay. See [build and reproduction instructions](tools/native_prf/README.md).

The native prototype is a promising way to reduce the measured Python/crypto
call overhead, but the previous 1.057928 upper serving ratio remains a failure.
Before any promotion: qualify a portable binary distribution, bind the native
implementation identity, verify full-caller sampling/lifecycle parity, and freeze
that runtime for a new declared serving study. No benchmark threshold or release
gate changed. No larger serving run was triggered from a helper result alone.

### Packaged native execution and confirmation

After the development screen, a separate unpublished `keyprint-native` wheel
integrates the helper through `execution="experimental-native"`. It statically
links OpenSSL 3.6.4, verifies installed-file hashes before loading, and binds the
binary, wrapper and build manifest into its own execution identity. The core
installation and default execution are unchanged. The macOS 11 ARM64 target is
verified from the library and static archive; actual execution was tested on
macOS 26.2 ARM64 only. A clean native-only environment needs no compiler,
Homebrew or separate crypto installation. See [installation](native/README.md).

Twelve complete reference/native Qwen caller pairs match exactly: 24 outputs,
988 committed tokens, full sampling records, diagnostics, text and consumed
work. Both actual local provider clients pass cancellation after a committed
token, terminal replay without regeneration, worker reuse and graceful shutdown.
A fresh native-only installation passes wheel RECORD and full-digest checks.
The 25 native boundary checks pass against the bundled library; the same C
source also passes the 10,000-call address/undefined-behavior sanitizer harness
linked to the new static OpenSSL. OpenSSL itself was not sanitizer-instrumented.

Only after those checks, the frozen runtime completed a new confirmation using
the same eight deterministically selected Dolly prompts, four repeats per
prompt, retained warmups, bootstrap and unchanged 1.05 threshold:

| Measure | Native execution |
| --- | ---: |
| Marked/ordinary seconds per committed token, geometric mean | 1.028758 |
| One-sided 95% bootstrap upper ratio | 1.032226 |
| Whole-request latency ratio, geometric mean | 0.997144 |
| One-sided 95% whole-request upper ratio | 1.024519 |
| Measured outputs / retained warmups | 64 / 16 |
| Audited committed tokens including warmups | 10,634 |
| Measured outputs ending at the token cap | 38 |

The local incremental timing screen passes. All attempts remain retained;
there were no retries, trimming or post-result exclusions. Independent audit
reconciles the declared prompts, bound runtime, artifact hashes, journal chains,
every displayed byte and token, and the primary ratio and bootstrap summary.
The comparison page includes all 32 ordinary/marked pairs and all truncations.
It does not approve their meaning or quality.

No other active inference or benchmark was observed at preflight. Idle preview
servers remained, and macOS Spotlight indexing was active and recorded. This
is not an isolated operating-system measurement. The approximate bootstrap
describes these fixed prompts only; the different generated lengths and paths
remain part of the study. The ordinary arm is the same SDK, so passing this
screen does not establish total overhead against an unmodified inference
server, batching throughput, HTTP/streaming cost, A18 or production acceptance.
The prior Python-runtime failure remains retained at its original identity.

```sh
python tools/serving_confirmation.py --model PINNED_MODEL \
  --corpus PINNED_DOLLY_JSONL --execution experimental-native \
  --expected-runtime FROZEN_RUNTIME_SHA256 --idle-host-confirmed \
  --output NEW_PRIVATE_CONFIRMATION
python tools/audit_serving_confirmation.py --run NEW_PRIVATE_CONFIRMATION \
  --model PINNED_MODEL --corpus PINNED_DOLLY_JSONL
```

### Unmodified MLX-LM baseline

The next declared study compares unmodified `mlx_lm.stream_generate` with
ordinary and marked `experimental-native` Keyprint, sharing one verified
Qwen3-8B model and tokenizer. All paths use the same six fixed prompts, caps,
temperature 0.7 and top-k 100. Execution order rotates across the three paths;
four repeats per prompt retain 72 outputs plus three declared warmups.

These are complete developer-facing paths, not identical sampling laws.
MLX-LM uses its own device sampler, special-token policy and renderer. Keyprint
uses its bound binary64 policy, cryptographic draws and durable reports/journals.
Both token denominators include EOS. Prompt encoding and rendering are timed;
model loading, imports, benchmark-artifact writing, HTTP and batching are not.
Keyprint's own durable artifact writing stays inside its measured generate call.

| Time per token comparison | Geometric mean ratio | One-sided 95% upper |
| --- | ---: | ---: |
| Ordinary Keyprint / MLX-LM | 1.485376 | 1.500379 |
| Marked Keyprint / MLX-LM | 1.526571 | 1.537920 |
| Marked / ordinary Keyprint | 1.027734 | 1.034310 |

Thus most of the observed excess is present even without marking. The prior
within-SDK pass does not establish negligible SDK overhead. All requests
completed without engineering errors. Audit reconciles 3,053 tokens including
warmups, replays the upstream renderer for its 995 tokens, and checks SDK
journal chains, prompts, byte rendering, bound source and statistical arithmetic.
A deliberately corrupted upstream token count is rejected by the auditor.
Every measured text appears in the resulting three-column comparison page.

The machine had active Spotlight indexing and other background processes;
no other active inference or benchmark was observed. This is approximate
fixed-workload evidence on one host, not operating-system isolation, production
server throughput, a same-distribution causal estimate, semantic approval or A18
acceptance. No output was retried, replaced or removed after measurement.

```sh
python tools/benchmark_engine_baseline.py --model PINNED_MODEL \
  --output NEW_PRIVATE_BASELINE --idle-host-confirmed
python tools/audit_engine_baseline.py --run NEW_PRIVATE_BASELINE
```

A separate cProfile run retains twelve more outputs and 502 tokens. Stable
support filtering consumes 3.94 instrumented seconds, about 7.9 ms per token;
instrumentation is not a serving measurement. This directs the next optimization
at full-vocabulary selection, not at weakening journals or masking output errors.

`tools/partition_topk.py` is a development-only selector, not SDK code. It finds
the cutoff, keeps every strictly higher score, chooses the lowest token IDs at
the cutoff tie, then sorts only the selected set. Validated candidate IDs remain
ascending and score arithmetic is unchanged. Forty boundary/property cases
match the reference order, including ties, signed zero, extreme float32 values,
full vocabulary and support smaller than k. Thirteen engine-harness tests also
pass. Ninety full-width helper comparisons preserve every selected position;
median candidate/reference selection-time ratios are 0.1064 for random logits,
0.2729 for rounded ties and 0.1986 for equal logits. The first equal-logit
slowdown is retained; the final candidate adds an exact all-equal shortcut.

Those were selector timings only. The following integration binds its own
source and qualifies the complete path; the 1.526571 result remains retained
for the preceding runtime.

### Integrated partition filtering

`experimental-native` now uses the partition selector inside a separately bound
full-filter implementation. The frozen engine is unchanged. Input validation,
float64 gap arithmetic, tie-breaking, immutable filtered logits, diagnostics and
the reference policy identity are preserved; the execution identity additionally
binds `partition_filter.py`. Per-step filter receipts and both pre-commit and
post-commit failure accounting remain intact. Journals are not weakened.

Forty-four full-filter cases compare exact output bytes and complete diagnostics
across extreme temperatures, rounded gap boundaries, ties, signed zeros, unmapped
padding and invalid inputs. Eighteen native checks pass alongside them. Four
additional failure tests preserve consumed-work behavior, including invalid
UTF-8 after commit. The installed wheel passes 798 regression tests with twelve
unchanged optional native-helper checks skipped; all 80 SDK files match source
and installation. Full-caller parity again reconciles twelve pairs and 988
reference/native tokens. Both actual client cancellation/reuse flows pass, as do
eleven JSON/typed-client requests with 213 tokens and 175 replayed grammar masks.

Only after qualification, the unchanged three-path benchmark ran against the
new frozen runtime. All 72 measured outputs and three warmups completed; all
measured outputs reached EOS. The audit reconciles 3,058 tokens, including 995
upstream engine tokens, and replays native rendering and SDK byte receipts.

| Time per token comparison | Geometric mean ratio | One-sided 95% upper |
| --- | ---: | ---: |
| Ordinary Keyprint / MLX-LM | 1.237856 | 1.245453 |
| Marked Keyprint / MLX-LM | 1.281173 | 1.289455 |
| Marked / ordinary Keyprint | 1.034994 | 1.041076 |

The complete SDK cost screen still fails. Within-SDK marking passes, but this
does not close the larger gap. The same prompt set, repeat count, order, threshold
and retention rules were used, with independent SDK draws and retained native
policies. These are separate local experiments, not identical generated paths or
an isolated causal estimate. OS background activity remains recorded, and no
batching, HTTP, semantic, detector or production acceptance follows. Further
optimization must preserve the same receipts and receive its own qualification.

The follow-up instrumented profile retains twelve outputs and 494 tokens.
Complete support filtering now accounts for 0.83 cumulative seconds, about
1.67 ms/token under cProfile. Hashing and repeated binding construction remain
visible costs; these diagnostics do not justify dropping integrity checks or
weakening durable journals. The uninstrumented complete-path result above is
the serving evidence.

### Reusing validated setup

The native candidate now prepares unkeyed tokenizer/profile metadata once.
Every request still hashes the pinned model/tokenizer files, checks immutable
binding fields and verifies channel markers. Replacement or corruption fails
closed. Mutable carriers, key state, random draws and journals remain local to
each request. The frozen constructor stays unchanged; the new implementation
has a separately bound source identity and no process-wide cache or patch.

Twenty new tests cover constructor-state equivalence, routing, concurrent
requests and changed profiles, configurations, mappings, files and markers.
The fresh installed wheel passes 822 tests, with twelve unchanged optional
development-library checks skipped. All 81 SDK payload files match source.
Actual caller parity again reconciles twelve pairs and 988 tokens; both provider
cancellation/reuse flows pass. Eleven structured requests reconcile 213 tokens
and 175 grammar masks, including typed clients and an explicit incomplete cap.

After qualification, the unchanged three-path study completes 72 measured
outputs and three warmups, with 3,065 audited tokens including 995 engine tokens.

| Time per token comparison | Geometric mean ratio | One-sided 95% upper |
| --- | ---: | ---: |
| Ordinary Keyprint / MLX-LM | 1.116311 | 1.126252 |
| Marked Keyprint / MLX-LM | 1.150419 | 1.158634 |
| Marked / ordinary Keyprint | 1.030555 | 1.034643 |

The total SDK screen still fails. Candidate creation and model loading remain
outside the per-request measurement, as before: this change amortizes validated
setup and does not establish lower cold-start cost. Marked/engine request
latency is 1.203121, versus 1.150419 per token, because generated lengths differ.
Separate SDK draws, native sampling policies, retained bookkeeping and active
OS background work prevent an identical-path causal interpretation. No timing
result closes quality, detection, A18 or production-serving gates. Every attempt
and earlier cost result remains retained.

### Bounded native selection

Accelerator `0.1.0a2` adds a bounded heap for finite binary32 scores. It retains
at most 512 entries and sorts only the retained set. Integer ordering preserves
subnormals, signed-zero ties and ascending token IDs. The SDK uses it only for
at least 1,024 candidates and k at most 512; equal scores, smaller inputs and
larger k retain the existing exact NumPy paths. Probability arithmetic and
dense SHA-256 receipt formats are unchanged. Older accelerator wheels fail
before model loading rather than failing during generation.

The first helper screen exposed slower equal-score selection; that result is
retained. With the existing equal-score shortcut preserved, all ninety final
full-width orders match an independent lexsort oracle. Median helper-time
ratios versus partition selection are 0.1752 for normal scores, 0.2087 for
rounded ties and 0.9856 for equal scores. These are helper measurements only.

The fresh core/native installation passes 918 tests without skips. Native tests
cover finite bit patterns, extremes, subnormals, signed zeros, ascending and
descending data, byte bounds, invalid input, output canaries and concurrent
calls. Both Python and native filtering paths pass 44 full-filter cases.
ASan/UBSan stress completes 10,000 valid and 10,000 invalid calls with unaligned
input and output guards; this is not a leak check or instrumented OpenSSL audit.
An isolated accelerator-only installation checks both HMAC digests and selection
without other Python packages. An actual older wheel is rejected before loading
weights. Qwen full-caller parity again reconciles 988 tokens, both client
lifecycle checks pass, and eleven structured requests reconcile 213 tokens and
175 grammar masks.

The unchanged full study completes 72 measured outputs and three warmups, with
3,018 audited tokens including 995 engine tokens:

| Time per token comparison | Geometric mean ratio | One-sided 95% upper |
| --- | ---: | ---: |
| Ordinary Keyprint / MLX-LM | 1.099011 | 1.106781 |
| Marked Keyprint / MLX-LM | 1.137564 | 1.146169 |
| Marked / ordinary Keyprint | 1.035080 | 1.039791 |

The complete-path 5% screen still fails. Marked/engine request latency is
1.153182 (upper 1.173548). The preceding runtime's marked/engine token ratio
was 1.150419; the new point estimate is modestly lower. Separate draws, output
lengths, native sampling policies and OS background work prevent a clean causal
comparison between experiments. No retries, outlier exclusions, quality,
detection or production acceptance follow from these results.

### Native v2 lossless vector commitments

Native execution now commits to a canonical, lossless encoding of complete
probability and filtered-logit vectors. Every binary64 bit is preserved,
including signed zero and NaN payloads. Sparse mode omits only a declared exact
default; dense mode remains available when smaller. This changes the digest
contract, so native runtime, facade and report versions are explicitly v2 with
new commitment field names. Old dense hashes are never relabeled. Codec and
validator source hashes are part of the execution identity; default and frozen
reference receipts retain their legacy format.

A fresh installed core wheel passes **1,079 tests without skips**; all 83 package
files match source and the wheel. The suite exercises both independent and SDK
codecs, malformed/noncanonical packets, source binding, random-draw invariants,
channel routing and pre/post-commit failure accounting. The existing native
accelerator binary is unchanged.

Twelve actual Qwen caller pairs retain 24 outputs and 988 tokens. Test-only
observation independently encodes each actual vector and checks reconstruction;
the offline audit resolves packets into their original dense bytes before
comparing legacy probability hashes. Text, random draws, diagnostics, journals
and consumed work match. Both local client cancellation/replay/reuse checks pass.
Eleven structured requests reconcile 213 tokens and 175 grammar masks, including
typed clients and an explicit incomplete cap. None of these checks accepts
prose quality or detection.

After those checks, the unchanged uninstrumented three-path study completed
72 measured outputs and three retained warmups. Independent audit reconciles
2,972 tokens, including 995 engine tokens, all displayed texts, journal chains,
runtime identity and declared statistical calculations:

| Time per token comparison | Geometric mean ratio | One-sided 95% upper |
| --- | ---: | ---: |
| Ordinary Keyprint / MLX-LM | 1.051768 | 1.058711 |
| Marked Keyprint / MLX-LM | 1.093868 | 1.100527 |
| Marked / ordinary Keyprint | 1.040028 | 1.046392 |

The complete-path 5% screen **still fails**. Marked/engine request latency is
1.052930 (upper 1.070735). The preceding native v1 runtime measured 1.137564
marked/engine time per token. These separate studies use independent SDK draws
and have different output lengths; the difference is not an isolated causal
estimate. Background macOS indexing and unrelated applications were recorded
and left running. No retries, replacements or outlier exclusions were used.
Imports, loading, HTTP and batching remain outside this measurement. The helper
speedup is not substituted for the full-path result, and A18 remains unaccepted.

### Full-gap range check

The experimental native filter now computes a conservative outward-rounded
bound on the complete finite score range and its scaled gap. If both bounds
pass, every finite mapped score must be admitted; only selected top-k scores
need per-token subtraction/division. Otherwise it uses the original calculation.
Caller-selected underflow warnings/errors also force the original path. Input
validation, exact output bytes, tie order, diagnostics and immutable backing
remain unchanged. See [the argument and reproduction steps](tools/FILTER_EXECUTION.md).

The installed wheel passes **1,153 tests without skips**, including 162 filter
checks. All 83 package files match source and the measured installation. A helper
screen retains 180 full-filter comparisons with the frozen reference; median
new/preceding time ratios range from 0.4438 for equal heads to 0.9609 for tight
gaps, including 0.5233 for normal full-width heads. These ratios exclude inference.

Actual caller parity again reconciles 988 tokens across twelve pairs, with
resolved probability bytes, random draws, text and consumed work matching.
Both provider lifecycle checks pass. Eleven structured requests reconcile
233 tokens and 190 grammar masks. After these checks, the unchanged full study
retains 72 outputs and three warmups and audits 3,041 tokens, including 995
engine tokens:

| Time per token comparison | Geometric mean ratio | One-sided 95% upper |
| --- | ---: | ---: |
| Ordinary Keyprint / MLX-LM | 1.061634 | 1.076050 |
| Marked Keyprint / MLX-LM | 1.108214 | 1.118923 |
| Marked / ordinary Keyprint | 1.043876 | 1.051788 |

Both the total-cost and within-SDK 5% screens **fail**. Marked/engine request
latency is 1.116217 (upper 1.143634). This did not improve the preceding 1.093868
marked/engine token ratio. The engine arm itself also slowed, from 26.317 to
28.454 ms/token geometrically averaged over the fixed cases. Background
applications were active and recorded. Their causal contribution is not
established, and the failed outcome is not discarded or replaced.

A separate diagnostic probe read actual filter frame locals without changing
filter/model functions. All 497 observed steps across twelve outputs used the
range shortcut; saved hashes, reports and journals reconcile. This rules out an
unused shortcut in that probe, but supplies no serving-speed or acceptance
claim. A better-controlled comparison is needed before calling this an SDK
performance improvement. No quality, detection, cost or release gate closes.

### Native digest decoding

A fresh profile of the preceding runtime retains twelve outputs and 501 tokens.
It identifies repeated Python label/layer loops in native digest decoding. The
native path now decodes batches of at least 32 labels with NumPy byte views,
while retaining scalar decoding for small batches. Label-major/layer-major
order, Python integer bits, duplicate-label handling and the full native HMAC
bytes stay unchanged. The native binary, default reference execution, sampling
law and durable journal policy are unchanged; the execution source hash changes.

Eighteen decoder/threshold cases plus the existing native suite pass. An
isolated wheel target using the existing dependency runtime passes **1,314 tests
without skips**; all 83 package files match source, wheel and installation.
This is not a fresh dependency installation or cross-platform qualification.
Twelve actual Qwen caller pairs reconcile 24 outputs and 988 tokens, including
exact text, random draws, probability bytes and consumed work. Both local client
protocols pass cancellation/replay/reuse checks. Eleven structured requests
reconcile 213 tokens and 175 grammar masks. These are engineering checks, not
quality approval or hosted GPT/Claude sampler integration.

Decoder-only measurements retain seven alternating-order pairs per setting.
For the default 30 layers, candidate/scalar median ratios are 0.1486 at 32
labels, 0.1310 at 100 and 0.1261 at 1,000. One-label and 31-label cases retain
ratios near one. These numbers exclude HMAC, model inference and serving.

The unchanged full-path workload retains all 72 outputs and three warmups. Its
independent audit reconciles 3,072 tokens, including 995 upstream-engine tokens,
all text/rendering receipts, SDK journals and statistical arithmetic:

| Time per token comparison | Geometric mean ratio | One-sided 95% upper |
| --- | ---: | ---: |
| Ordinary Keyprint / MLX-LM | 1.052950 | 1.058760 |
| Marked Keyprint / MLX-LM | 1.072251 | 1.081295 |
| Marked / ordinary Keyprint | 1.018330 | 1.024416 |

The complete-path 5% screen still **fails**. Marking-only cost passes this small
within-SDK screen. Marked/engine whole-request latency is 1.174492 (upper
1.194106), retaining differing generated lengths. Unrelated VM and desktop
workloads stayed running; no other model/test workload was active during timing.
There were no replacements, retries or outlier exclusions. This observation
does not isolate a causal speedup or establish production serving acceptance.

### Support-mask reuse

The shared MLX caller now counts finite raw logits without allocating their full
index array unless public-fixture capture explicitly needs those indices. Fast
and native hosts reuse the prepared support mask for their support check and
sampling lookup. Batched sessions reuse one output support mask for ordered
mass, equality checks and commit, and avoid copying an already-owned probability
snapshot on identity branches. Full-vector finite/negative/mass/support guards,
immutable prepared bytes, random draws and journal events remain intact.
The shared caller change is bound into all three current MLX execution modes;
the archived reference implementation is unchanged.

Twenty-one new checks cover sparse/dense/padded head receipts, explicit fixture
indices, input mutation, immutable prepared data, repeated/startup/protected
contexts and corrupted transform outputs. The installed candidate passes
**1,335 Python tests with no skips**. All 83 package files match source and wheel;
dependencies reuse the existing pinned environment.

Each of bounded reference, experimental-fast and experimental-native passes
twelve actual Qwen comparison pairs against the archived implementation. The
three independent artifact audits reconcile **72 outputs and 2,964 tokens**
across the same six fixed prompts, with matching text, probability bytes, random
draws and consumed work. Both local client lifecycle checks pass. Eleven native
structured requests reconcile 213 tokens and 175 grammar masks. An initial
runner resolved the interpreter outside its environment and failed to import
NumPy before model loading or output creation; that failed setup remains saved.

The unchanged uninstrumented workload and independent audit retain 72 measured
outputs, three warmups and 3,000 tokens, including 995 upstream-engine tokens:

| Time per token comparison | Geometric mean ratio | One-sided 95% upper |
| --- | ---: | ---: |
| Ordinary Keyprint / MLX-LM | 1.047557 | 1.055344 |
| Marked Keyprint / MLX-LM | 1.084027 | 1.091838 |
| Marked / ordinary Keyprint | 1.034814 | 1.039046 |

Both engine-relative 5% screens **fail**. The marked/engine observation is worse
than the preceding 1.072251 result, so this is not a demonstrated full-path
speedup. Whole-request marked/engine latency is 1.091026 (upper 1.108037).
All outputs and lengths remain included. Background VM and desktop activity
continued; no other task-owned inference or test workload ran during timing.
These separate samples do not isolate the code's causal timing effect. Fewer
allocations and exact parity do not close the serving-cost gate. A controlled
whole-path comparison is needed before attributing an improvement or regression.

### Immutable raw-head snapshots

The experimental native caller and shared filter previously hashed the same
complete float32 model head independently. They now share one immutable byte
snapshot and its computed SHA-256 digest. No caller-supplied digest is accepted.
Standalone supplied-array steps retain their original hash path. Grammar
masking creates a new snapshot: pre-mask and post-mask journal commitments stay
distinct. Numeric validation, immutable probability outputs, journal durability,
random-draw ordering, sampling and report formats remain unchanged. Source hashes
bind the new execution; the frozen implementation and native binary are unchanged.

Sixteen new checks cover mutation, signed zero/subnormal/NaN bit preservation,
strided inputs, invalid shapes, filter rejection before randomness, raw-hash
reuse and grammar commitments. The installed wheel passes **1,351 tests with
zero skips**. All 83 package files match source, wheel and installation. Each
of the three MLX modes also passes twelve actual Qwen caller comparisons against
the frozen implementation, totaling 72 outputs and 2,964 tokens across the same
six prompts. Both local client lifecycle checks pass. Eleven native structured
requests reconcile 218 tokens and 175 grammar masks.

The new [matched-revision procedure](INFERENCE_TESTING.md#matched-path-revision-timing)
keeps two independent models resident and runs sequential, alternating-order
requests. The preceding support-mask revision and the new candidate receive the
same public fixture key, prompts and explicit random draws. All 96 measured
requests and four warmups remain retained. An independent audit reconciles
4,144 tokens, exact text, sampling records, consumed work, fixture random draws
and journal events apart from the expected runtime/source identities.

| Candidate / preceding revision | Geometric mean ratio | One-sided 95% upper |
| --- | ---: | ---: |
| Ordinary caller time per token | 0.995297 | 1.003657 |
| Marked caller time per token | 0.994067 | 1.000761 |

Token paths and lengths match, so request-latency ratios equal the per-token
ratios. The roughly 0.5–0.6% point-estimate reductions are **not established
speedups**: both upper bounds include no improvement. No observations were
excluded or retried. Other task-owned tests/inference finished before timing;
unrelated VM/desktop workloads remained. Fixture randomness, two resident
models, omitted public `generate()` preflight and absent OS isolation limit
extrapolation. This is not a fresh engine-relative cost measurement and cannot
close the 5% serving gate. Twenty separate benchmark/auditor checks pass.

Initial snapshot tests looked for private filter details in a public report;
the corrected tests capture the private receipt before projection. The first
independent audit compared a full worker identity to the report's compact target
and rejected it. The corrected auditor verifies the full specification digest
before checking its documented compact projection. Both initial failures remain
saved; no inference or timing results were replaced.

## Remaining qualification

- Qualify real SGLang/vLLM request lifecycles using this new adapter source.
- Profile isolated end-to-end ordinary and marked serving, including journals.
- Measure more model/tokenizer families and realistic batch sizes.
- Establish held-out calibration, quality and short-text power.

The playground exposes per-run generation and inspection times and real work
stages. It deliberately retains weak measurements. In two saved reference runs,
English generation-path counts matched literal replay exactly (1,291/2,490).
French generation-path counts were 772/1,560 versus literal replay 775/1,560.
Replay differences therefore did not explain that run's weak signal; longer,
controlled experiments are still required. No detection threshold or acceptance
was added from these observations.
