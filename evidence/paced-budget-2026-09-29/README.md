# Spread the probability budget across more layers

The first bounded oracle spent its probability budget quickly. A separately
identified exact-rational policy now limits each layer's absolute change to
one eighth of each token's original probability, while retaining the original
half-to-double total bounds. The SDK and its default sampler remain unchanged.

The same four existing two-token fixtures were evaluated at thirty layers.
The one-eighth policy, fixtures and source hashes were recorded before running
the comparison. There was no parameter sweep, sampled-state pruning or selection
of favorable results. Every fair independent binary-label assignment is covered
through exact rational state propagation.

| Original probabilities | Previous expected bit agreement | Paced expected bit agreement | Previous positive-strength layers | Paced positive-strength layers |
| --- | ---: | ---: | ---: | ---: |
| 50% / 50% | 50.833333% | 52.259415% | 2.000 | 21.690 |
| 75% / 25% | 50.833185% | 51.331778% | 3.999 | 25.570 |
| 99% / 1% | 50.033327% | 50.053271% | 3.999 | 25.570 |
| 99.9% / 0.1% | 50.003333% | 50.005327% | 3.999 | 25.570 |

The ordinary or independent-key expected agreement is 50% under these idealized
assumptions. The improvement does not measure detector power: observations may
be correlated, real vocabularies are larger, deployed keys are reused, and
language-model paths change. Positive strength can still produce no change when
labels agree. No text has been generated or repaired in this comparison.

## Construction and verification

Let `p` be the initial distribution and `q` the current state. Retain the previous
oracle's common safe strength, then further bound it by
`(p_i / 8) / (q_i * (1 - q_i))` for every `0 < q_i < 1`.
Choose this strength before reading the next layer's bits; apply the same update
`q'_i = q_i * (1 + alpha * (g_i - sum_j(q_j * g_j)))`.

For any binary labels, `|g_i - sum_j(q_j * g_j)| <= 1 - q_i`, so each layer
changes token `i` by at most `p_i / 8`. The original safe-strength constraint
simultaneously retains `p_i / 2 <= q'_i <= 2 * p_i`. Total mass and support are
preserved exactly. With fair next-layer labels independent of previous layers,
the conditional expected change is zero. Grouped labels retain within-group
ratios. None of these facts implies unchanged semantics or a fixed-key sequence
distribution.

The one-eighth value is an exploratory choice, not a calibrated operating point.
It prevents the lower half-mass allowance from being spent in fewer than four
maximal downward steps from the initial distribution. It cannot prevent eventual
absorption: after thirty layers the paced policy has exhausted strength in
60.76% of balanced-fixture label histories and 34.53% of the other three fixtures.

Twenty-nine focused checks pass, including the earlier twenty oracle/capacity
checks and nine new paced-policy checks. The new tests exhaust 1,536 three-layer
paths, compare the final-token score with the conditional-expectation method,
check grouped/zero support and adverse labels, and independently reproduce the
balanced thirty-layer result using an absorbing integer random walk. Previous
maximum-step results still match every stored value. Unknown policies fail.

The comparison completed under a 256 MiB external watchdog. The sampled process
footprint was about 10.2 MiB; cleanup succeeded. No model inference or GPU work
ran, and no previous replay attempt was resumed or overwritten.

## Next qualification

This justifies further numerical investigation of pacing, not SDK promotion.
Remaining work: finite-precision verification, realistic grouped distributions,
source/EOS protection and repeated-context behavior, then actual paired inference,
frozen factual/language review, and independent matching-/wrong-key calibration.
The original replay remains 111/128 complete pending memory headroom. Quality,
full clue conformance and launch readiness are not established by this result.

```sh
PYTHONPATH=src:tools python -m pytest -q \
  tests/test_paced_budget_oracle.py \
  tests/test_predictable_budget_capacity.py tests/test_predictable_budget_oracle.py
python tools/compare_paced_budget.py --output /path/to/new-comparison-directory
```

Recorded [plan](plan.json), [exact results](results.json) and
[resource/source validation](validation.json).
