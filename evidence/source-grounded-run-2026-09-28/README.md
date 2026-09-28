# Source-grounded paired inference: in progress

The 128-attempt run has started. No factual-quality result, detector qualification
or launch acceptance is available from this study yet. All prior multilingual
and expansion-task failures remain; this adds a natural source-based task family.

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

## Verification snapshot

[Partial receipt audit](partial-audit.json) verifies the first 45 completed
attempts and 8,642 committed tokens, with no execution errors in that snapshot.
This is not a full-schedule audit or independent model-forward replay. It checks
native bytes, prompt IDs, usage, EOS/cap behavior, journal hash chains, RNG rejection
transcripts and commit order. No semantic labels or condition-level counts were
read to prepare the review machinery.

[Local checks](local-checks.json): 55 tests pass. They reject missing, reordered
or duplicate attempts; changed review text, sources or rubrics; leaked condition
metadata; modified native rendering/prompts/usage; and incomplete ratings. Tests
also keep token caps in the denominator, separate coverage from supported claims,
apply sensitivity only to explicitly pre-flagged uncertain fields, and treat
unavailable detection measurements as unavailable rather than negative results.

## Review after inference

`tools/audit_source_grounded.py` requires all 128 scheduled attempts and the
complete metadata-hidden review. It verifies source/rubric and SDK commitments,
native receipts, model asset pins and every blinded view. Error attempts remain
visible with hashes of any partial receipts, never counted as verified success.
Key slots are planned assignments; this receipt audit is not an independent
re-execution of the keyed model path.

`tools/build_source_review.py` builds a private offline page with each original
passage, task, output and criteria. Human ratings start blank. Coverage, claims,
language and format stay separate; exported partial or unsure answers never
become acceptance. This new page has unit coverage; rendered browser QA is still
pending until the actual completed study is available. The production landing
page is unchanged.

Assistant ratings must be frozen against shuffled outputs before joining key,
condition or count metadata. `tools/summarize_source_grounded.py` then reports
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
