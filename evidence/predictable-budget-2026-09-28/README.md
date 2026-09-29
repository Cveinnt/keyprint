# Predictable probability budget: exact-rational oracle only

Status: mathematical prototype, not an SDK path, selected inference candidate,
accepted watermark or quality fix. The 128-path diagnosis remains incomplete
after a resource interruption; its original evidence is unchanged.
No model output, detector result or clue acceptance follows from this oracle.

## Motivation and distinction

The current tournament can strongly concentrate conditional probabilities for a
particular key. The prior five-layer and half-mixture studies did not qualify a
replacement. Reducing average concentration does not explicitly bound how much
an individually unlikely token can be amplified. The new oracle explores a
per-token likelihood-ratio bound instead.

[The LUNA paper, Section 4.3](https://arxiv.org/html/2606.00613v1#S4.SS3)
explicitly distinguishes random-key single-token marginal preservation from
fixed-key and full-sequence equality. Its theorem is context for this distinction,
not evidence for this proposed budget, implementation or quality. No novelty or
private Anthropic implementation claim is made.

## Exact construction

Let p be the initial categorical distribution and q the current layer state.
Keep each q_i within [p_i / C, C p_i]. This oracle fixes C = 2 for mathematical
checks; that value is not an empirically selected quality/detection operating point.

Before reading the next layer's bits, choose a common alpha in [0, 1] bounded by
both (C p_i - q_i) / (q_i (1 - q_i)) and
(q_i - p_i / C) / (q_i (1 - q_i)) for every 0 < q_i < 1.
Then apply q'_i = q_i [1 + alpha (g_i - sum_j q_j g_j)].

Any binary assignment changes q_i by at most alpha q_i (1 - q_i), so the entire
update stays inside the specified bounds. Zero entries remain zero. Exact mass
sums to one. Shared group bits retain within-group ratios. With fresh fair next-layer
bits, alpha is fixed given prior layers, and E[q'_i | prior layers] = q_i.
Iteration therefore preserves p in expectation under this idealized per-prefix
randomness model. This is not a theorem about reused deployed keys, arbitrary
sequence distributions, finite-precision code or semantic accuracy.

Choosing alpha before seeing the next bits matters. Clipping a completed proposal
and then renormalizing does not generally preserve the expectation. An exact
counterexample for p = (3/5, 2/5) is included in the tests.

## Verification and unresolved costs

Twelve local checks pass. Three distributions each exhaust every binary assignment
across three layers and three labels: 1,536 complete paths. Exact Fraction arithmetic
checks normalization, all bounds and equality of the averaged output with the input.
Other checks cover shared labels, bit-independent step size, zero support, invalid
states, a clipping counterexample and thirty layers favoring a rare token.

A token initially at probability 1/1000 never exceeds 2/1000 in the thirty-layer
favorable-bit test. Once a bound is reached, the common alpha can become zero and
all remaining layers contribute no new watermark evidence. This is a material
capacity risk, not a successful detector result. The oracle neither changes
post-generation text nor identifies which tokens carry facts or language.

The [exact capacity analysis](../predictable-budget-capacity-2026-09-29/README.md)
now quantifies early strength exhaustion in four two-token mathematical fixtures.
It does not qualify this oracle or establish model-level detector performance.

Before any promotion: complete and inspect the guarded diagnosis; decide whether to
build a separately identified floating-point candidate; verify numerical/support,
EOS and source-protection behavior; and run actual ordinary/reference/candidate
inference with every attempt retained. Frozen factual review, multilingual tests,
matching-key power and independent ordinary/wrong-key calibration must be assessed
together. Old detector thresholds and prior clue acceptance cannot transfer.
