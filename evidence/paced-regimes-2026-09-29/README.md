# Where the bounded candidate carries information

All 128 retained paths and 24,421 token positions were inspected without new
inference. Five entropy bins were declared before this diagnostic ran; they are
descriptive, not tuned detector thresholds. All 12,462 marked positions reconcile
with the previous complete conditional-information diagnostic. The 11,959
ordinary positions describe actual ordinary distributions, not marked
counterfactuals. EOS and every low-information position remain included.

| Native base entropy, bits | Marked positions | Share of positions | Conditional KL sum, nats | Share of KL |
| --- | ---: | ---: | ---: | ---: |
| Below 0.1 | 7,318 | 58.72% | 0.216746 | 0.63% |
| 0.1 to below 0.5 | 1,700 | 13.64% | 2.401066 | 7.02% |
| 0.5 to below 1 | 1,432 | 11.49% | 8.014269 | 23.44% |
| 1 to below 2 | 1,596 | 12.81% | 16.884074 | 49.39% |
| At least 2 | 416 | 3.34% | 6.672161 | 19.52% |

Positions with at least 0.5 bits of native base entropy account for 27.64% of
positions and 92.34% of measured conditional information. Across all marked
positions, 7,234 (58.05%) have a base maximum probability of at least 99%.
The candidate's recorded kernel froze at least one layer in 2,270 positions.
Of 18,884 frozen layer calls, 18,579 (98.38%) occur below 0.5 bits. The previous
synthetic comparison also failed to improve signal by changing boundary
allocation alone. Together, these observations argue against treating boundary
freezes as the primary demonstrated fix. They do not establish causality.

These KL sums are conditional quantities along realized opened paths, not a
population information bound, calibrated detection power or quality measure.
Native base entropy depends on the original prompt and model state. An arbitrary
text inspector does not possess that state. Selecting these positions with an
oracle would not establish an accessible detector, and changing the statistic
would require fresh calibration. Entropy also does not certify that alternatives
preserve facts or meaning.

## Earlier filtering experiment is not an untried fix

The September 19 prompt-free detector-side filter used a fixed continuation
prefix, model inference and a 0.5-bit selection rule on a different profile and
opened cohort. Its fixed-batch study flagged 10/12 marked responses versus 8/12
for its baseline; 3/5 short marked responses versus 1/5. All 12 ordinary and the
wrong-key comparisons remained unflagged in that small development set.

The larger null run was **incomplete**: 500 planned controls, 499 usable and one
failed control. On the same usable controls, the filter flagged eight versus
five for the baseline. The retained partial audit recomputed 998 scores but did
not accept the full run or production calibration. The missing control is not
a negative. These historical thresholds and results cannot transfer to the
current candidate. Relevant current receipt hashes and summaries are retained
in [the reconciliation](prior-filter-reconciliation.json); no model replay of
that historical filter was performed in this turn.

## Decision and validation

Do not ship an entropy filter or launch another identical 30-layer study on
these results. Next compare signal efficiency of separately specified bounded
generation mechanisms on saved distributions, including the information-rich
positions. Require a justified key-averaged distribution argument, exact
probability/excluded-mass checks and an accessible scoring design before new
inference. Existing per-layer pacing is a research design choice, not a published
Anthropic requirement. Changing it still defines a new candidate requiring fresh
semantic and detection validation; total probability bounds alone are not a
semantic guarantee. Keep the frozen quality failures and release hold.

Thirteen tests cover entropy-bin edges, invalid values, exact fixture metrics,
ordinary/marked separation, changed hashes, incomplete paths and EOS handling.
Every progress row and aggregate was checked; prior information helpers and all
SDK source hashes remain unchanged. The run completed in 4.52 seconds at
82,919,952 bytes (79.08 MiB) peak under a 512 MiB watchdog. All sixteen pressure
samples were normal; cleanup verified. No model loading or GPU inference.

[Plan](plan.json), [all results](results.json), [validation](validation.json).
