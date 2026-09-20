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
