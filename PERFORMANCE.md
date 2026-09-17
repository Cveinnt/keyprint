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
