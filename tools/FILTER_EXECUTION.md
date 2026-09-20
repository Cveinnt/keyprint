# Exact filter execution

Experimental native execution preserves the frozen shared support policy and
records a separate implementation identity. This optimization changes neither
the watermark transform nor the admitted token law.

## When the range check is sufficient

For finite mapped binary32 scores converted exactly to binary64, let `M` and `m`
be their maximum and minimum. The reference computes each rounded loss
`d = M - score`, admits it through the existing coarse raw-gap bound, then checks
that its rounded quotient `d / temperature` is finite and no greater than the
declared gap. Temperature is finite and positive. The score range cannot
overflow binary64.

The shortcut rounds `M - m` upward with `nextafter`, divides that bound by
temperature, then rounds the quotient upward again. Monotonic subtraction and
division ensure these bounds are at least every reference loss and quotient.
If both bounds pass the original coarse and final gap checks, every finite
mapped score is admitted. Top-k can then run immediately on those scores; only
the selected scores need the original subtraction/division. No reassociation
or approximate softmax is introduced.

When either bound fails, execution uses the original per-token gap calculation.
It also uses that path whenever NumPy underflow handling is not `ignore`, so
top-k selection cannot hide a warning, callback or error from an excluded token.
The original head validation still rejects NaN and positive infinity, including
unmapped padding. Input bytes, token-ID tie order, gap/top-k counts, policy
identity and immutable output are preserved.

## Reproduce

The complete-filter suite compares output bytes, IDs, diagnostics, identities,
immutability and exceptions with the frozen reference. It covers arbitrary
finite bit patterns, signed zeros, subnormals, extreme temperatures, rounded gap
neighbors, padding, strided inputs and caller underflow policy.

```sh
python -m pytest -q tests/test_partition_filter.py
python tools/benchmark_gap_filter.py --baseline PRECEDING_FILTER_SOURCE \
  --candidate src/keyprint/experimental/partition_filter.py --output NEW_SCREEN
```

The September 20 helper screen retains 180 full-width comparisons with the
reference. Median new/preceding filter-time ratios are 0.5233 for normal heads,
0.5225 for rounded ties, 0.4438 for equal scores, 0.8251 for sparse heads,
0.9609 for tight gaps and 0.5349 for very large temperature. Validation,
diagnostics and immutable output are inside timing; file writing and oracle
checks are outside. All repetitions are retained without trimming.

These are complete-filter helper measurements, not full SDK cost or production
acceptance. Actual caller parity, lifecycle tests and the unchanged uninstrumented
engine comparison are documented in [performance evidence](../PERFORMANCE.md).
The subsequent full-path study measured 1.108214 marked/engine time per token
(upper 1.118923), worse than the preceding 1.093868 result. It remains failed.
A diagnostic `tools/probe_filter_branches.py` run observes the shortcut on all
497 actual steps across twelve outputs without patching the filter or model.
The helper result and branch coverage do not establish an end-to-end speedup.
Hosted CI remains disabled; qualification runs locally.
