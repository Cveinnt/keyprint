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
This low-level factory is not wired into default `Keyprint.generate`, the CLI or
the HTTP service. Pipeline construction still creates a fresh bound candidate;
startup and request reuse need measurement before production integration.

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
installed wheel's complete local suite passed 276 tests. Full caller lifecycle,
end-to-end serving measurements and exact-revision hosted checks are still
required. The earlier 7.6% incremental cost and failed 5% screen describe the
unchanged default path; this parity result does not replace those measurements.

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
