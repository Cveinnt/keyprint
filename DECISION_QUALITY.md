# Decision accuracy under ordinary and marked generation

All **128 outputs answered correctly** in a fixed synthetic decision screen:
64 ordinary and 64 marked. This result covers constrained JSON decisions on
pinned Qwen3-8B/MLX reference execution. It does not resolve retained free-form
rewrite failures or qualify general production quality.

## What ran

Sixteen prompts form eight positive/negative pairs in English, Spanish, French
and Chinese. Each changes a fact that changes the answer: approval versus
acknowledgment, the authorized approver, verified restoration, or approval before
a deadline. Each prompt runs four times per condition. Order alternates by case
and repetition; draws are independent. Ordinary uses the same SDK filtering and
grammar as marked, with watermarking disabled.

The grammar permits both `{"allowed": true}` and `{"allowed": false}`. It
constrains format, not the answer. Correctness requires EOS and exactly the
expected boolean. Truncation, duplicate fields, numeric substitutes, commentary
and extra fields fail. No request was retried or replaced. The prompts, answers,
schema and all-pass rule were frozen before generation.

| Condition | Correct | Incorrect | Runtime failures |
| --- | ---: | ---: | ---: |
| Ordinary | 64 | 0 | 0 |
| Marked | 64 | 0 | 0 |

An independent receipt audit reconciles 128 responses, 1,047 committed tokens
and 1,047 grammar masks. It checks journal chains, exact prompt hashes, assigned
conditions, returned text, completion, usage and recomputed answers. Of 528
marked sampling records, 288 have different base/prepared weight hashes. That
bookkeeping observation is not an effect-size or detectable-mark guarantee.
Model logits were not replayed.

## Limits

These are sixteen repeated synthetic prompts under one key, not 128 independent
topics or a representative deployment sample. There is no blind human review,
causal harm estimate, prose-fidelity acceptance, language-retention test or
detector-power claim. The constrained response has very little free text.
Known free-form copying, unwanted translation and approval-condition drift
remain recorded in other quality studies. Output quality and launch remain open.

This distinguishes success on explicit decisions from failures when rewriting
surrounding prose. It does not establish why those rewrite failures occur or
that watermarking never contributes to them.

## Reproduce and inspect

Install the SDK with `mlx,structured` extras and use pinned
`mlx-community/Qwen3-8B-4bit` revision
`545dc4251c05440727734bcd94334791f6ab0192`:

```sh
python tools/validate_decision_quality.py \
  --model models/qwen3-8b-4bit --output private-decision-quality
```

The command writes its plan before generation, retains every attempt, and exits
unsuccessfully if any answer fails. `public/comparison.html` shows every prompt
and exact output. Private root folders contain the key and journals; do not
publish them. New runs use fresh randomness and may differ.

The reviewed run used SDK revision `c21e4f36dfa598e7b745a44e62f782003a42160c`.
Public-only evidence: [plan](evidence/decision-quality-2026-09-20/plan.json),
[all outputs](evidence/decision-quality-2026-09-20/results.json), and
[receipt audit](evidence/decision-quality-2026-09-20/integrity.json).
Fourteen validator tests exercise false positives, missing/duplicate attempts
and balanced answer pairs.
