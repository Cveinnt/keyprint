# Fixed prose comparison, September 23

This small actual-inference pilot had mixed results. Ordinary outputs passed
10/12 frozen task rubrics; marked outputs passed 8/12. All 24 outputs stayed in
the requested language and completed without runtime errors. This is descriptive
evidence, not quality acceptance or a general estimate of watermark harm.

| Frozen criterion | Ordinary | Marked |
| --- | ---: | ---: |
| Picnic invitation without invented date, time, location or names | 2/4 | 4/4 |
| Project email preserving names, deadline and approval condition | 4/4 | 4/4 |
| Two French sentences explaining preferential blue-light scattering | 4/4 | 0/4 |
| All task criteria | 10/12 | 8/12 |
| Requested language | 12/12 | 12/12 |

The two picnic failures invented a park. The four marked French outputs repeated
the same answer, saying blue light scatters in all directions without explaining
preferential scattering or shorter blue wavelengths. They are French, coherent
answers, but fail the stricter explanation rubric fixed before generation. They
are not translation failures. Treating them as four independent topic-level
failures would overstate this three-prompt experiment.

## Method and retained evidence

- Qwen3-8B-4bit revision `545dc4251c05440727734bcd94334791f6ab0192`,
  MLX reference backend, temperature 0.7, top-k 100, cap 192 tokens.
- Three prompts, four repetitions each, ordinary and marked; one fresh private
  key. Conditions alternate first position. No retries, replacement samples,
  output rewriting or constrained decoding.
- All 24 actual outputs are retained in [results.json](results.json).
  Every token trace reconstructs its rendered text exactly.
- The assistant rated randomized condition-hidden outputs against
  [fixed rubrics](plan.json), then saved [ratings](frozen-ratings.json) and
  [hash commitments](rating-commitment.json) before joining conditions.
  This was not independent human review.
- Six pairs passed both conditions, two only marked, four only ordinary.
  Observed marked-minus-ordinary failure rate: +16.7 percentage points.
  Reused prompts and one key do not support a general causal or noninferiority
  conclusion. No scientific gate is closed by this pilot.

## Repetition investigation

[Freshness audit](freshness-audit.json): all 24 attempts recorded model calls and
distinct random-draw transcripts. Their committed token counts reconcile.
Identical marked French text therefore was not simply a cached response in this
run. This audit does not establish random-number statistical quality or a cause
of reduced output diversity.

| Distinct full outputs in four repetitions | Ordinary | Marked |
| --- | ---: | ---: |
| Picnic | 4 | 2 |
| Approval | 3 | 2 |
| French | 4 | 1 |

Next quality investigation: a separately declared multi-key repetition test,
retaining every output. Do not change the sampler, rerate this batch or choose
favorable keys to make this result disappear. The SDK's research engine is
unchanged by this audit.

## Reproduce and inspect

Run `tools/prose_quality_pilot.py` with the exact cached model revision above and
a new output directory. Review `public/blind-review.json` before opening the
private condition mapping; use `summarize()` to join frozen ratings. The
registered script and case hashes are in `plan.json`. Fresh sampling will not
reproduce the same text byte for byte. Private key and raw draw journals are
retained locally, excluded from this evidence export.

`sha256.json` covers the six exported JSON receipts. Public samples contain only
synthetic prompts and generated text. No claim that the SDK preserves all meaning
or never produces an unwanted translation follows from this run.
