# Sampling performance

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
