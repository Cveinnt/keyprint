# Complement-symmetric allocation: verified mathematics, no promotion

The new exact rule retains the existing half-to-double total probability bounds and one-eighth-of-base per-layer movement budget. It uses the actual label split rather than the previous worst-case bound. This is a separately identified research policy, not an SDK change or inherited detector calibration.

## Exact argument

For base probabilities p, current probabilities q and binary labels g, let G = sum(q_i g_i) and d_i = q_i(g_i - G). Choose alpha at most one and at most each available two-sided probability/movement radius divided by abs(d_i), ignoring zero movements. Apply q'_i = q_i + alpha d_i.

Total mass stays one because sum(d_i) is zero. The radius constraints preserve p_i/2 <= q'_i <= 2p_i and abs(q'_i-q_i) <= p_i/8. Under complemented labels, d changes sign while every radius and alpha stay unchanged. The two complementary outputs therefore average exactly to q. For complement-symmetric next-layer labels independent of previous layers, conditional expectation is unchanged. Shared label groups retain within-group probability ratios.

This is a different proof from choosing strength before observing bits. It does not justify arbitrary label-dependent clipping. It also does not prove exact marginal preservation for finite PRF keys, fixed deployed keys or rounded implementations, and does not guarantee semantic preservation.

## Numerical verification and retained failure

The first floating-arithmetic version failed two of forty fixtures: wide-tail/100 weights, seeds 0 and 1, at layers 22 and 27 respectively. Independently reproduced base-relative trajectory differences were 6.50e-9 and 2.11e-10, exceeding the unchanged 1e-10 tolerance. Intermediate rounding caused different later boundary-freeze decisions. Both failed records, original source and plan are retained here.

The corrected policy, `complement-symmetric-exact-proposal-eighth-v1`, selects a rational strength using integer comparisons, computes the rational proposal with integer products, and rounds once to binary64. Existing exact normalized-categorical checks still reject budget or support violations. There is no clipping or fallback.

All **40/40 fixtures and 1,200 updates** now match the independent Fraction proposal/reference trajectories exactly. Fifteen tests include all 1,536 exact three-layer paths, grouped ratios, boundary states, complemented-label equality, extreme-tail regressions and a rounded two-layer marginal check over all 64 label paths. The finite numerical checks do not upgrade the exact ideal-label theorem into a universal floating-point unbiasedness claim.

## Signal comparison

Fixtures use 100 and 1,000 weights, four fixed seeds, thirty shared label layers and all prior 32 numerical cases plus eight declared 99%-peaked cases. The table reports mean expected bit-score lift over the ordinary distribution under the **same finite label arrays**, in percentage points. It does not assume each finite fixture baseline is exactly 50%.

| Fixture family | Previous paced lift | Complement-symmetric lift |
| --- | ---: | ---: |
| uniform | 1.849872 | 1.323675 |
| zipf | 1.624849 | 1.184478 |
| wide-tail | 0.704434 | 0.479569 |
| grouped | 1.816690 | 1.338467 |
| peaked | 0.028799 | 0.028614 |

Bit-score lift falls in every fixture family. The highly peaked fixtures have mean conditional KL of 0.000316 nats before versus 0.000327 after, a small absolute change. Higher information in uniform/grouped fixtures comes with more total variation and lower uniform-score lift. These are synthetic counterfactual probabilities, not text quality or calibrated detection results.

**Do not promote this variant or spend a model run on it as a presumed fix.** It does not demonstrate resolution of the low-information issue seen in real outputs. Preserve the result and separate mathematical validity from practical usefulness.

Next inspect where probability-budget capacity exists in recorded prefixes and what is lost to low predictability. Reopen the earlier prompt-free entropy-filter work before proposing another filter, so a new generation policy is not confused with already-tried detector-side selection. Future proposals need separately declared profiles and fresh semantic/detection confirmation.

The corrected run completed in 50.16 seconds under a 512 MiB watchdog, with 79,168,040 bytes (75.50 MiB) peak footprint, 181 normal-pressure samples and verified cleanup. No model inference or GPU work. The initial run also cleaned up; its failure was numerical validation, not a memory cutoff.

[Corrected plan](plan.json), [results](results.json), [validation](validation.json), [initial failed results](initial-results.json), [failure reproduction](initial-failure-reproduction.json), [original source](initial-first-attempt-source.py.txt).
