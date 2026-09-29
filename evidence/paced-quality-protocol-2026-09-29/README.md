# Candidate source-quality review protocol

This protocol was prepared while the candidate inference study was running,
before reviewing its source texts or joining factual ratings to conditions and
signal counts. **No candidate quality ratings or acceptance are recorded here.**
The source tasks, factual rubrics and metric definitions remain those of the
earlier source-grounded study; no easier criterion replaces them.

Every one of the 128 attempts needs exactly one explicit source-only rating:
each essential fact, unsupported claims, language and requested format, plus an
explanation. Uncertain judgments must be flagged while their strict value is
false. Execution and decoder failures remain in the denominator with no passing
or uncertain components. A capped output may have correct content but cannot
pass the complete-task metric.

`tools/summarize_paced_quality.py freeze` validates the ratings against the
blinded packet and copies their original bytes into a fresh private directory.
It binds the complete cohort, source/rubric content, review packet and summary
code by hash, without joining ratings to conditions or computing signal counts.
Existing freezes cannot be overwritten. A file commitment records this workflow;
it cannot prove what a reviewer previously saw.

A separate `summarize` invocation requires that commitment and unchanged inputs.
It reuses the existing source-study metric functions for strict results, an
all-preflagged-uncertainties sensitivity analysis, paired counts, and per-source
and per-key breakdowns. It includes raw token-path counts without a detector
threshold. It cannot turn a decoder failure into a successful text result.

These are descriptive assistant development judgments on sixteen reused English
source tasks and four reused keys. They are not independent human acceptance,
a powered causal/noninferiority comparison, detection calibration or a universal
language/meaning guarantee. Earlier multilingual and factual failures remain.

## Verification and usage

Seventeen new quality-workflow tests pass; **43 checks** pass when combined with
the fifteen cohort checks and eleven existing source-metric checks. Tests cover
freeze-before-join behavior, failure accounting, preserved uncertainty, unchanged
denominators, incomplete ratings, assignment metadata, edited ratings/evidence
and changed helper code. Synthetic ratings in tests are not actual study ratings.

After the complete run is audited and every blinded output reviewed:

```sh
PYTHONPATH=src:tools python tools/summarize_paced_quality.py freeze /private/paced-study \
  --original /private/original-source-run --ratings /private/source-only-ratings.json \
  --frozen /private/new-rating-freeze
PYTHONPATH=src:tools python tools/summarize_paced_quality.py summarize /private/paced-study \
  --original /private/original-source-run --frozen /private/new-rating-freeze \
  --output /private/new-quality-summary.json
```

The tool identifies these as assistant development ratings. Human review stays
separate and is never inferred from permission to proceed or from automated tests.

Recorded [protocol and code bindings](protocol.json). SDK defaults, original
ratings and public release status remain unchanged.
