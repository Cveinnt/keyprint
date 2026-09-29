# Complete paced-candidate source review

All 128 outputs received explicit assistant source-only judgments before condition/key/count joins. The freeze binds ratings, original rubrics, all generation evidence and unchanged metric helpers. All 64 uncertainty fields were recorded before unblinding. Private source text, derived outputs and individual rating reasons remain outside this repository. This is assistant development review, not independent human acceptance.

| Measure | Ordinary / 64 | Marked / 64 |
| --- | ---: | ---: |
| All essential facts, strict | 36 | 35 |
| No unsupported claims, strict | 38 | 31 |
| Both content checks, strict | 25 | 18 |
| English | 64 | 64 |
| Requested format, strict | 63 | 59 |
| Full task, strict | 25 | 16 |
| Both content checks, all preflagged uncertainty accepted | 36 | 34 |
| Full task, all preflagged uncertainty accepted | 36 | 34 |

Strict full-task paired counts are 12 both pass, 13 ordinary-only, 4 marked-only and 35 neither. These are paired prompts, not shared random draws. Sixteen reused tasks and four reused keys create dependence; this is not a powered noninferiority result. The uncertainty sensitivity is not a confidence interval and does not replace strict ratings. Source-fact composites are demanding, and ambiguous omissions or wording were flagged explicitly. All attempts remain included.

Observed failures include invented dates and institutions, altered attribution, omitted conditions, head-to-head versus overall statistics, and unsupported clinical efficacy. They occur in both conditions. All outputs remained English in this English-only cohort; that is not multilingual qualification or a guarantee against translation.

## Signal and promotion decision

Raw token-path matching rates were 49.9160% ordinary versus 50.1190% marked under the matching keys. Next-key rates were 49.9746% and 49.8835%. These correlated layer/token counts are uncalibrated diagnostics, not detection rates, p-values, independent Bernoulli trials or arbitrary-text detector evidence. The observed matching-key separation is only 0.2030 percentage points.

The candidate is **not qualified for SDK promotion or launch**. Neither negligible quality impact nor useful calibrated detection has been established. Exact probability bounds alone do not imply semantic equivalence. Low absolute ordinary quality also prevents treating all output failures as watermark-caused.

Do not tune or rerate this cohort to create a pass. Preserve this development result. Next work must separate base-model fidelity, watermark quality impact and detectable signal; use a prospectively frozen, disjoint evaluation before any release claim. Human review remains available separately. No post-generation rewriting, retries, omitted failures or forced translations are permitted.

[Full aggregate results](results.json) include per-case/per-key counts and strict/sensitivity paired outcomes. [Freeze commitment](commitment.json) binds private judgments without publishing source-derived text. Candidate generation and resource receipts remain in [paced-native-run](../paced-native-run-2026-09-29/README.md).
