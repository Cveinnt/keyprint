# Signal cost of the first bounded-probability oracle

The exact-rational oracle prevents an individual token from moving outside
half to twice its original probability. It chooses the largest currently safe
common strength before reading each layer's bits. That policy can spend its
entire probability budget early, leaving later layers unmarked.

We now quantify this weakness for four fixed two-token mathematical fixtures.
These are exact state-propagation results, not sampled estimates or language
model outputs. All thirty layers and all possible independent fair binary label
assignments are included. The SDK and default sampler are unchanged.

| Original probabilities | Expected matching-bit fraction | Expected layers with positive strength | Probability strength exhausted by layer 30 |
| --- | ---: | ---: | ---: |
| 50% / 50% | 50.833333% | 2.000000 | 99.9999999% |
| 75% / 25% | 50.833185% | 3.999286 | 99.982142% |
| 99% / 1% | 50.033327% | 3.999286 | 99.982142% |
| 99.9% / 0.1% | 50.003333% | 3.999286 | 99.982142% |

The ordinary or independent-key expected matching-bit fraction is 50% in this
idealized model. A layer with positive strength can still have no effect when
both tokens receive the same bit. Positive-strength layers therefore do not
mean independent useful observations.

## Exact method and cross-checks

Propagate probability mass over the oracle's exact rational state after each
layer. No low-probability states are discarded. Stop with an error if the fixed
state budget is exceeded. At every layer, verify that state masses sum exactly
to one and that the weighted mean distribution equals the original distribution.

The score concerns the **final sampled token's** bits across all thirty layers.
Given past labels, future predictable oracle updates preserve each token's
probability in expectation. Consequently the expected final-token agreement
with an earlier layer equals its expectation immediately after that layer.
This conditional-expectation identity allows accumulation of the final score
without enumerating all complete label histories or drawing tokens.

Three independent three-layer exhaustive enumerations check that identity for
two-token and three-token distributions. They compute final token probabilities
first, then score every earlier label. A separate thirty-layer closed-form
check covers the equal-probability case: the first differing labels move the
state to (3/4, 1/4) or its reverse and exhaust the common strength. The expected
matching fraction is exactly `1/2 + (1 - 2^-30) / 120`.

Twenty focused checks pass, including the original twelve oracle checks and
eight capacity-analysis checks. The actual analysis completed under a 256 MiB
external watchdog; only about 10.3 MiB appeared in its sampled footprint. No
GPU, model load, generated output, secret key or detector calibration was used.

## Decision and limits

Do not promote this maximum-safe-step oracle as a quality fix. It retains its
exact probability bounds and ideal marginal-preservation result, but these
examples demonstrate early loss of marking strength. They do **not** establish
detector power, false-positive rates, factual preservation, model-vocabulary
behavior, grouped-label behavior, or failure of every bounded approach.

The next candidate design must address how quickly it spends the probability
budget, rather than silently loosening the bounds. Any slower allocation policy
needs its own numerical/oracle checks and retained actual inference comparisons.
No fixed-key sequence guarantee or Anthropic implementation claim follows.

Reproduce from the repo root:

```sh
PYTHONPATH=src:tools python -m pytest -q \
  tests/test_predictable_budget_oracle.py tests/test_predictable_budget_capacity.py
python tools/predictable_budget_capacity.py --output /path/to/new-results.json
```

Exact fractions and per-layer state counts are in [results.json](results.json).
Source commitments and the resource receipt are in [validation.json](validation.json).
