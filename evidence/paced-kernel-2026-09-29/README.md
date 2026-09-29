# Faster paced kernel with exact categorical checks

The new research kernel replaces Fraction operations in the hot path with
integer arithmetic for strength selection and bound checks, and binary64
arithmetic for proposals. It remains outside the SDK. No model inference,
semantic-quality result, detector calibration or production acceptance follows.

## Numerical policy and independence

The policy retains total probability ratios in `[1/2, 2]` and per-layer changes
of at most one eighth of original probability. It reserves an absolute margin
of `base_probability * 2^-40` inside the available upper, lower and movement
budgets before reading labels. A state with no reserved room freezes predictably.
This is a separately identified policy, not the previous reference's
multiplicative strength backoff. Earlier results remain retained separately.

The fast implementation uses exact integer ratios to choose strength, then rounds
that strength down to binary64. An independently written Fraction reference
computes the reserved margins directly in probability space and forms exact
proposals before output rounding. Comparisons cover both identical-prefix updates
and independent full thirty-layer numerical trajectories.

All output certifications operate on the probabilities actually sampled. If base,
previous and output weights have integer representations `(b,B)`, `(q,Q)` and
`(r,R)`, the checks use exact cross-products:

```text
2 * B * r >= b * R
B * r <= 2 * b * R
8 * B * abs(r * Q - q * R) <= b * R * Q
```

These retain the total and per-layer bounds without approximate normalization.
Support changes and violations raise errors. There is no clipping, flooring,
retry, redraw or alternate-output fallback. The admitted range remains 1–4,096
weights and positive normalized base probabilities of at least `2^-900`.

## Paired numerical study

All **32 fixed cases and 960 updates passed**, using the same uniform, Zipf-like,
600-log-unit-tail and unequal grouped fixtures at sizes 100 and 1,000, with four
seeds each. The plan and code hashes were saved before execution. No failures or
cases were discarded. Call order alternates each layer; only step calls are timed,
including their checks, excluding base setup and comparison work.

| Measurement | 100 weights | 1,000 weights |
| --- | ---: | ---: |
| Passed cases / updates | 16 / 480 | 16 / 480 |
| Kernel median of case-median layer times | 0.199 ms | 1.947 ms |
| Fraction reference, same statistic | 2.475 ms | 25.595 ms |
| Ratio of those medians | 12.41× | 13.15× |
| Largest base-relative trajectory difference | 5.13e-15 | 6.20e-15 |
| Kernel / reference near-boundary freezes | 5 / 5 | 0 / 0 |

The fixed comparison tolerance was `1e-10`; actual differences are shown above.
Numerical agreement is not bit-identical output or proof of unchanged semantic
behavior. The run was a single local microbenchmark with other processes active,
including focused tests during part of execution. It is not an isolated serving
benchmark. At the 100-weight median, thirty layers imply roughly 6 ms per token;
end-to-end model overhead remains unmeasured for this candidate.

The external 256 MiB guard recorded ~18.3 MiB sampled peak footprint, completed
normally and verified cleanup. No model or GPU was loaded.

## Verification and remaining work

Seventy-four focused checks pass: previous 53 mathematical/numerical checks plus
21 kernel/reference checks. New checks exhaust 1,536 three-layer paths, verify
strength never rounds above the exact value, compare thirty-layer trajectories,
and reject malformed inputs, changed support, escaped bounds, normalization traps
and invalid labels. These are not a fresh full-SDK suite.

Next: source/EOS/repeated-context integration under an explicit new profile,
followed by retained actual paired generation, factual/language review, independent
ordinary/wrong-key calibration and serving measurements. All 78 SDK Python sources
remain unchanged. The original diagnostic replay is still 111/128 pending memory
headroom; no interrupted run has been silently restarted.

```sh
PYTHONPATH=src:tools python -m pytest -q tests/test_paced_integer_kernel.py
PYTHONPATH=src:tools python tools/benchmark_paced_kernel.py --output /path/to/new-study
```

Recorded [plan](plan.json), [all cases and timing samples](results.json),
and [source/resource receipt](validation.json).
