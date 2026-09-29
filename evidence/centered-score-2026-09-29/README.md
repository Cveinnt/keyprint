# Exact bounded score allocation: saved-prefix comparison

These are research mechanisms, not SDK changes or production profiles. They use
the same thirty keyed layer labels as the existing candidate for a paired
comparison on all 64 already-opened marked paths. Every original prefix, EOS
step and repeated context is retained. No new text is sampled, rewritten,
translated, repaired or selected. The new conditional distributions do not
describe the trajectories these mechanisms would generate.

## Two separately evaluated rules

Both rules preserve support, normalization and excluded-token probabilities
exactly using integer categorical distributions. Eligible token probabilities
remain between half and one-and-a-half times their base probabilities. They
replace thirty sequential p/8-paced adjustments with one aggregate adjustment;
they do not inherit the per-layer movement budget, profile identity or detector
calibration. No entropy filter is applied.

**Centered linear score.** Let s be a token's sum of thirty keyed bits and let
mu be the base-weighted mean within eligible tokens. Set
q_i = p_i [1 + (s_i - mu)/(2 max_j |s_j - mu|)]. If all scores agree, use identity.
The centered term sums to zero. Complementing every bit sends s to 30-s and
negates the centered term without changing its scale. The two distributions
therefore average exactly to p. The integer implementation has no intermediate
floating-point proposal or clipping.

**Balanced score allocation.** Start each eligible token at p_i/2, then allocate
the remaining half of eligible probability mass in descending score order,
with capacity p_i for each token. Tokens tied at the boundary share allocation
proportionally to base mass. This maximizes expected supplied score subject to
normalization, excluded probabilities and the stated [p/2,3p/2] box. If positive
allocation remains at a lower score while a higher score has spare capacity,
moving probability upward increases the objective; this exchange argument
establishes the greedy optimum. It does not optimize KL, semantics or detection
power, nor establish an optimum over the older asymmetric [p/2,2p] region.

Complemented scores reverse upper and lower half-mass allocations. Proportional
tie handling makes those allocations add to each token's original mass, so
the two final distributions again average exactly to p. Thus either rule has
unchanged conditional mean under an ideal complement-symmetric fresh label
ensemble. This is not exact finite-PRF or deployed-fixed-key unbiasedness,
guaranteed language/meaning preservation, or a guarantee against factual errors.

## Scoring and experimental limits

The supplied score is the mean of keyed bits, computable from a key and a
canonical token/context path without native probabilities. The comparison
reports its expected lift over the ordinary distribution under the same label
array. It does not assume a finite-label ordinary baseline is exactly 50%.
Native probabilities are used to measure conditional KL and total variation,
not to make an entropy-based text selection rule.

An accessible mathematical statistic is not yet a calibrated arbitrary-text
detector. A deployable profile still needs separate domain binding, round-trip
text behavior, held-out null controls, fixed-key behavior and quality/power
tests on newly generated outputs. Same old labels here isolate a conditional
rule comparison; they must not be silently reused as a new generation profile.

The initial linear comparison's exact runner source is retained in
`initial-comparison-source.py.txt`. Only after that job completed was the runner
extended with an explicit balanced option; its new plan and output directory are
separate. Both results and all rows remain retained. See adjacent plans, results
and validation for measurements and the resulting decision.

## Complete comparison and decision

Each new rule was evaluated on all 12,462 saved prefixes from the same 64 marked
responses. Old conditional KL and total variation reconcile against the prior
complete diagnostic. Values below are sums along each recorded path, averaged
over the 64 paths. They are not population KL bounds or response-level TV.

| Metric | Previous paced rule | Centered linear | Balanced allocation |
| --- | ---: | ---: | ---: |
| Sum of conditional KL(q/p), nats | 0.534192 | 0.342029 | 2.930209 |
| Sum of conditional total variation | 2.672407 | 2.104153 | 7.170201 |
| Sum of expected mean-bit-score lift | 0.399323 | 0.365378 | 0.889522 |

The centered linear rule moves less probability but supplies less absolute
signal. It is not a detection fix. Balanced allocation supplies 2.23 times the
expected score lift and 5.49 times conditional information, with 2.68 times
the summed probability movement. Its stronger signal therefore comes with a
material distortion tradeoff, despite tighter per-token ratio bounds. It is
not an improvement at matched distortion, nor evidence of better generated
text or detection power. Its exact score optimum applies only to its declared
box and supplied linear objective.

This warrants a separately bound experimental profile and a prospectively
frozen, disjoint paired generation study, not SDK promotion. First preserve
exact excluded-mass handling, integer draws, canonical contexts and accessible
score replay under the new profile. Then evaluate every generated output for
facts, language, conditions and task coverage, alongside fresh null controls
and detection. Retain all errors and truncations. Never tune or rerate the old
128-output quality cohort into acceptance. The publication hold remains.

Thirty tests pass, including 1,536 exact three-layer label paths for each rule,
independent Fraction references, tied-score/exclusion checks, and 64 exhaustive
small score assignments compared against every feasible box/simplex vertex.
Runner checks cover repeated contexts, EOS, changed hashes, incomplete paths
and rejection of untested source-protection modes. All results, progress rows,
aggregates and bound source hashes were verified. SDK sources remain unchanged.

Both offline jobs completed under separate 1 GiB guards with normal pressure and
verified cleanup. Linear: 64.82 seconds, 445.17 MiB peak, 233 pressure samples.
Balanced: 67.55 seconds, 437.33 MiB peak, 241 pressure samples. Only tokenizers
loaded; no model inference. Initial linear runner source remains retained.

[Linear plan](centered-plan.json), [linear results](centered-results.json),
[balanced plan](balanced-plan.json), [balanced results](balanced-results.json),
[validation](validation.json).
