# Source-to-launch claim audit

The historical 22/25 ledger counts scoped project criteria. It is not proof that
Keyprint satisfies every behavioral claim in Anthropic's announcement. Preserve
the original receipts and criteria; do not silently upgrade their meaning or
transfer them to Qwen3.5 experimental-wide or a proposed new sampler.

## Public source, checked September 28

[Anthropic's announcement](https://www.anthropic.com/news/claude-text-watermark)
identifies its method as a SynthID-Text variant. It describes choices that preserve
quality and meaning, avoid implausible alternatives, and offer less watermarking
capacity where factual or code correctness constrains the answer. These behavioral
claims are broader than numerical support preservation. Its serving statement is
qualitative; it does not specify our historical 5% timing threshold. Its translation
example concerns requested translation, not translating an answer to add a watermark.

The named algorithm family is public. Reproducing related behavior does not prove
recovery of Anthropic's private implementation, key, parameters or detector.

## What our evidence establishes

| Ledger item | Retained scoped evidence | Unsupported launch inference |
| --- | --- | --- |
| A10, model support | Post-filter positive alternatives remain reachable | Their probabilities remain close, or unlikely factual alternatives cannot be amplified |
| A15, factual sparsity | A fixed learned-head panel measured lower average conditional perturbation for factual prompts; a confident-error counterexample remains | Facts in arbitrary generated responses are preserved, or all fact-bearing positions receive no perturbation |
| A17, exact code | Supplied singleton logits preserve a forced answer; a fixture allows valid comment alternatives | Real models reliably keep executable code correct when several tokens have positive probability |

The A10/A15/A17 descriptions above were checked against the existing ledger's
acceptance criteria and limitations. None of these scoped results becomes a new
failure merely because it is narrow. The error would be presenting it as evidence
for the broader behavior in the right column. The current 128-output review adds
relevant task-level evidence; it does not erase earlier multilingual/code limits.

## Consequences for the next candidate and launch

The live original-path replay measures probability concentration and selected-token
base likelihood. It can distinguish a numerical mismatch from a correctly executed
but unsuitable sampling law. It cannot alone attribute factual mistakes to that law.

The exact probability-budget oracle addresses one narrower mechanism: amplification
relative to the original probabilities. A model can assign substantial probability
to a false answer, so that bound is not a semantic safeguard. It may also reduce
watermark capacity. Do not advertise it before native inference and signal validation.

Keep candidate-quality comparisons, calibration, integration/lifecycle qualification
and product/demo readiness explicit. No zero-error base-model requirement replaces
the original comparative quality target. No clipped or rewritten output repairs the
experiment. A launch description must name the tested configuration, retain failures
and state detection limits. No 25/25, universally correct, uniquely identified,
hosted-provider injection or production-ready claim is supported here.
