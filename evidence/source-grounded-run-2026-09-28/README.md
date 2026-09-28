# Source-grounded paired inference: factual review complete

All 128 planned outputs completed at EOS, with no execution errors or truncations.
All 24,465 sampled tokens pass the complete receipt audit. All source-only
assistant judgments and uncertainty flags were frozen before joining condition,
key and count metadata. Content passes are 33/64 ordinary and 23/64 marked;
accepting every pre-flagged ambiguity gives 43/64 and 37/64. This does not establish
quality acceptance or a causal effect. All prior multilingual and
expansion-task failures remain; this adds a natural source-based task family.

## Frozen factual review

| Outcome | Ordinary / 64 | Marked / 64 |
| --- | ---: | ---: |
| All required source facts covered | 47 | 36 |
| No unsupported claims | 39 | 38 |
| Both content checks | 33 | 23 |
| English retained | 64 | 64 |
| Original requested format | 63 | 63 |
| Full task | 32 | 22 |
| Content, all pre-flagged ambiguities accepted | 43 | 37 |
| Full task, all pre-flagged ambiguities accepted | 42 | 36 |

The strict content comparison has 16 pairs passing both, 17 ordinary-only,
7 marked-only and 24 passing neither. Sensitivity gives 30, 13, 7 and 14,
respectively. These are 16 tasks and four reused keys, not 64 independent task
clusters or a powered noninferiority study. The between-arm difference is
descriptive; it does not isolate a causal watermark effect or replace the older
registered quality criterion.

The larger difference is in required fact coverage, while unsupported-claim
counts are similar. Sixty uncertain fields were recorded before unblinding,
including whether brief answers needed a secondary job title, retirement detail,
pre-Act treaty context or subnational geography. Accepting those fields changes
the size of the difference but does not reverse its direction. No rating was
changed after seeing the arm totals. Correct-looking summaries can omit required
facts; supported claims and completeness therefore remain separate metrics.

Clear errors remain in both conditions. Marked examples include a winged
"Wingless Victory", applying Finland's support percentage to Sweden too,
misassigning an album's distribution history, and changing an incident's timing
and number of officers who fired. An ordinary output misassigns a multi-cause
mortality projection to extreme weather alone; another denies prior founder
wealth despite the supplied explanation. These are source-faithfulness findings,
not current-world fact checking or proof of a systematic causal mechanism.

[Complete aggregate results](fidelity-results.json) include every case/key and
both sensitivity definitions. [Final rating commitment](rating-commitment.json)
binds the plan, blinded inputs and all private ratings. An [independent count
check](summary-check.json) reproduces the strict and sensitivity totals without
calling the summary implementation. Detailed judgments and source-derived text
remain in private receipts; human ratings remain separate and blank.

Raw matching counts are 176,105/352,200 ordinary and 173,184/335,580 marked
(50.0014% and 51.6074%). Next-key counts are 175,853/352,200 and
168,025/335,580 (49.9299% and 50.0700%). All 128 inspections were available.
These are correlated diagnostics, not independently sampled bits, calibrated
detector accuracy, false-positive estimates or evidence of Claude authorship.
No old-profile threshold is borrowed.

## Frozen before generation

- Sixteen instructions and their source passages from the separately pinned
  [selection](../source-grounded-selection-2026-09-28/README.md).
- Sixteen source-only rubrics, sixty essential fact checks, original task-format
  requirements and qualifiers. [Commitment](rubrics-commitment.json) binds both
  private case and rubric files. Optional background detail is not a mandatory
  fact; unsupported claims and missing essential facts are scored separately.
- [Run plan](plan.json): every ordinary/marked pair across all four existing
  keys, alternating condition order, Qwen3.5 experimental-wide, temperature 0.7,
  top-k 100, cap 768. No output selection, retries, rewriting or replacement.
- [Identity](identity.json): pinned model, assets, native runtime, tokenizer and
  sampling profile. SDK sources remain unchanged during this run.

## Complete receipt verification

[Complete receipt audit](receipt-audit.json) verifies all 128 attempts and all
24,465 committed tokens: native bytes, prompt IDs, usage, EOS/cap behavior,
journal hash chains, RNG rejection transcripts and commit order. It also binds
each blinded view to the exact original source, frozen rubric and recorded output.
All 128 draw transcripts are distinct. This is not independent model-forward or
keyed-source re-execution, semantic acceptance or detector calibration. The
[earlier partial snapshot](partial-audit.json) remains retained.

[Local checks](local-checks.json): 55 tests pass. They reject missing, reordered
or duplicate attempts; changed review text, sources or rubrics; leaked condition
metadata; modified native rendering/prompts/usage; and incomplete ratings. Tests
also keep token caps in the denominator, separate coverage from supported claims,
apply sensitivity only to explicitly pre-flagged uncertain fields, and treat
unavailable detection measurements as unavailable rather than negative results.

## Original-path diagnosis

The [frozen diagnostic plan](sampling-diagnosis-plan.json) covers all 128 original
paths and every step, regardless of rating, key or signal. Execution is underway;
no complete replay result or mechanism conclusion is claimed yet. Direct native
forwards, the full-head filter, scalar source policy and exact recorded draws
must match original receipts. No new outputs, draws, retries or repairs.

Telemetry measures conditional distribution concentration, selected-token base
probability/rank and EOS mass at the same recorded prefix. EOS mass and the
probability of reaching that prefix are different quantities; these diagnostics
cannot alone explain omissions or establish semantic causation. Between-arm
paths also differ. Both token-weighted and equal-output summaries are planned.
The summarizer requires the final result and all 128 verified paths; partial or
failed cohorts cannot become a success-only aggregate. All failures stay retained.

[Diagnostic checks](diagnostic-checks.json): 58 focused local checks pass, including
known probability distributions, unchanged telemetry inputs, impossible support,
missing/reordered/failed attempts, edited measurements and premature EOS. This
suite overlaps earlier research checks; counts are not added together. The SDK
and its default sampler remain unchanged during replay.

[Summary entrypoint follow-up](summary-entrypoint-checks.json): 16 checks pass,
including file-backed final-result commitments, refusal of progress-only and
failed results, and a variable-length fixture separating token weighting from
equal-output weighting. This overlaps the earlier suite and is not added to it.

## Review and execution tools

`tools/audit_source_grounded.py` requires all 128 scheduled attempts and the
complete metadata-hidden review. It verifies source/rubric and SDK commitments,
native receipts, model asset pins and every blinded view. Error attempts remain
visible with hashes of any partial receipts, never counted as verified success.
Key slots are planned assignments; this receipt audit is not an independent
re-execution of the keyed model path.

`tools/build_source_review.py` builds a private offline page with each original
passage, task, output and criteria. Human ratings start blank. Coverage, claims,
language and format stay separate; exported partial or unsure answers never
become acceptance. [Actual-data browser QA](review-qa.json) passes desktop
1440×1000 and mobile 390×844 checks: page identity, nonblank content, no framework
overlay/console errors/overflow, source visibility, hidden labels, blank initial
ratings, required format check, persistence, partial export, uncertainty as null,
and navigation through all 128 entries. QA actions remain confined to isolated
browser contexts and are not study ratings. The production landing page is unchanged.

The [partial-rating commitment](rating-progress.json) retains the first 64 manual
assistant judgments across eight complete cases; its hash was checked unchanged
before freezing all 128. Clear failures and interpretively uncertain fields were
recorded separately. No partial arm-level comparison was computed.
`tools/summarize_source_grounded.py` reports
all 64 pairs, by-case and by-key outcomes, and a sensitivity analysis limited to
the pre-flagged fields. Missing detector verdicts remain missing. No thresholds
from another profile or bitwise significance claims are used.

Source articles, prompts, derived outputs and detailed ratings remain in private
receipts, separate from the MIT SDK. Public artifacts contain provenance, hashes
and aggregate checks only. This English-only development sample is not known to
be held out from model training and is not a powered noninferiority study.

SDK publication, production deployment, hosted CI and publicity remain held.

## Reproduce a new run

Use the frozen selection/rubrics and original development keys locally. All
outputs differ under new random draws; do not relabel them as this study.

```sh
PYTHONPATH=src python tools/validate_source_grounded.py \
  --selection /private/frozen-selection --model /path/to/pinned-snapshot \
  --prior /private/prior-multikey-study --output /private/new-run
PYTHONPATH=src python tools/audit_source_grounded.py /private/new-run \
  --model /path/to/pinned-snapshot
python tools/build_source_review.py /private/new-run/private/blind-review.json \
  /private/new-run/private/review.html
```
