# Conditional information in the paced candidate

Every one of the 128 recorded paths and 24,421 token steps reconciles. The analysis reconstructs the marked conditional distribution on both ordinary and marked prefixes. It matches the saved marked distribution or ordinary base distribution exactly at every step, includes EOS, and retains every output. It does not sample new tokens or change any text.

| Diagnostic | Ordinary / 64 | Marked / 64 |
| --- | ---: | ---: |
| Mean path log(marked/base), nats | -0.66887 | 0.52241 |
| Median path log(marked/base), nats | -0.42580 | 0.41722 |
| Paths with likelihood ratio at least 100 | 0 | 0 |
| Mean sum of conditional KL(marked/base), nats | 0.55443 | 0.53419 |
| Mean sum of conditional KL(base/marked), nats | 0.55774 | 0.53828 |

The median marked path ratio is approximately 1.52:1. Along marked paths, average conditional total variation per step is about 0.01372 and conditional KL(marked/base) about 0.00274 nats. These observations support investigating insufficient per-response information rather than assuming a different display score alone will make the candidate useful.

## Interpretation limits

This is a privileged-probability oracle using saved native base weights, generation keys and original tokenizer paths. A public text detector does not ordinarily possess these inputs. This diagnostic is not the SDK detector and cannot be substituted for one.

The fixed 100:1 ratio cutoff was declared before computation. Its observed hit count is not a calibrated production detection rate. It is also not directly comparable to the different random-key rank cutoff in the earlier screen. No cutoff was selected after viewing results.

Conditional KL sums are realized compensators along these paths; their cohort averages are not proven population KL bounds. Sixteen reused tasks, four reused keys and opened development outputs do not establish powered noninferiority, a fixed-key false-positive rate, or an impossibility theorem for every detector. The calculation also does not attribute factual failures to watermarking.

The study used general-purpose mode with no source-copy protection. All prompt sources, original outputs, earlier failures and frozen quality ratings remain retained. A source-preservation feature or a numeric probability bound is not proof of semantic preservation.

## Validation and resources

Twelve new tests cover exact finite probability examples, adaptive likelihood normalization and expected-log-ratio identities, conservative input validation, near-identity calculations, and the distinction between ordinary stored distributions and marked counterfactuals. Counterfactual construction shares the frozen research kernel; no independent native model-head replay occurred here.

An independent Fraction-based calculation checks selected likelihood ratios for all 12,462 marked steps directly from original journals. Maximum per-path discrepancy is 1.56e-15. All 128 progress rows match final results and all result bindings verify.

Tokenizer only; no model weights or inference. The 2 GiB watchdog completed in 303.60 seconds at 452,608,960 bytes (0.42 GiB) peak sampled footprint. All 1,089 pressure samples were normal and cleanup verified. No automatic restart.

## Decision and next work

Do not promote the current paced candidate. Its frozen factual review remains unresolved and neither ordinary scoring nor this privileged diagnostic establishes useful single-response detection. Preserve the complete failed development result.

Next compare information allocation within the existing probability and excluded-mass constraints, using exact finite examples and saved distributions before another model run. Any proposed change must have a separately identified profile and an explicit argument for its key-averaged behavior, rather than simply increasing distortion. Fresh, prospectively frozen detection and semantic evaluations remain required; no tuning or rerating this cohort into acceptance.

[Plan](plan.json), [complete results](results.json), [validation](validation.json).
