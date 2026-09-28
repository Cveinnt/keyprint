# Longer output carries signal, but does not meet the preservation bar

The unchanged SDK generated **32 fresh responses** on four longer source-grounded
writing tasks, across four existing keys. All finished without runtime errors.
Eleven of sixteen marked outputs crossed the unchanged matching-key diagnostic
rule; **none passed fidelity, language, format and detection together**.

This is a failed development screen, not launch acceptance or proof that the
watermark caused the failures. Ordinary outputs also fail the strict content
rubric. The tasks deliberately revisit known difficult domains and require
expanding compact source facts without adding information. That combination
creates a floor effect and cannot estimate a causal watermark quality penalty.

| Measure | Ordinary | Marked |
| --- | ---: | ---: |
| Completed at EOS | 16/16 | 16/16 |
| Strict fact preservation with no unsupported additions | 0/16 | 0/16 |
| Content if all ten pre-flagged conservative judgments pass | 5/16 | 5/16 |
| Requested language, without foreign-word intrusion | 16/16 | 15/16 |
| Requested length and three prose paragraphs | 3/16 | 2/16 |
| Matching-key diagnostic hits | 0/16 | 11/16 |
| Next-slot-key diagnostic hits | 0/16 | 0/16 |
| Strict full-task and detector joint passes | 0/16 | 0/16 |
| Joint passes with all conservative judgments accepted | 0/16 | 0/16 |

The marked language failure is one English word, **neither**, inserted into
Spanish prose. No response translates wholesale into another language. This
still violates the requested language consistency and remains a failure.
Fifteen outputs in each arm have three paragraphs; length fails much more often.
Neither arm hits the generation cap. A format failure is reported separately
from a factual error, not relabeled as semantic damage.

## What actually failed

- Shipping responses add promises to notify the customer, keep them informed or
  maintain communication, despite explicit instructions not to invent them.
  One invents an assurance that order composition and shipping will not change.
- Maintenance responses invent system checks, restoration-test preparations,
  notification policies or future contact. All original operational facts often
  remain present, but the additions still violate the brief.
- French comparisons retain the base prices and capacities while adding wrong
  arithmetic: 20 GB becomes twice or five times 5 GB; another says B costs more
  per GB, despite 18/20 being less than 12/5. Some turn unspecified annual prices
  or contract terms into confirmed absence of those options.
- Handoffs add active-verification status, separate teams, exhaustive blocker
  claims, publication promises or absent schedules. Some are conservative
  interpretation boundaries; every such flag was frozen before unblinding.

Strict judgments are assistant labels, not independent human acceptance. Ten
boilerplate/status interpretations are flagged conservative in
[frozen-ratings.json](frozen-ratings.json). Accepting all ten improves content to
five passes in each arm, but produces no marked full-task or joint pass. Both
views are retained in [sensitivity.json](sensitivity.json); original labels are
unchanged. These counts should not be generalized to arbitrary writing quality.

## Reproduce and inspect

- [Every paired response and its review](SAMPLES.md).
- [Metadata-hidden review page](review.html): source facts, exact outputs, blank
  human ratings, local persistence and explicit export. No automatic acceptance.
- [Frozen plan](plan.json), [review commitment](rating-commitment.json),
  [results and token audits](results.json), [SDK integrity](sdk-integrity.json).

Pinned Qwen3-8B MLX revision
`545dc4251c05440727734bcd94334791f6ab0192`; temperature 0.7, top-k 100, 1,024-token
cap. Four reused keys, including known failures. Four tasks in Spanish, Japanese,
French and English, each ordinary/marked once per key. Alternating arm order;
no retries, replacements, text repairs, favorable-key selection or threshold
changes. Each Latin-script task requests 200–260 words and three prose
paragraphs; Japanese requests 400–600 Unicode characters and three paragraphs.
All whitespace counts toward the Japanese character measure.

The detector retains the weighted 30-layer rule at nominal tail <= 0.005 per
key, checking matching and next-slot keys. This is an uncalibrated development
measure. Zero observed wrong-key hits on this small set does not establish a
false-positive rate, robust attribution or deployment power. It is not the
historical 0.001 fiction-bank detector rule on the landing page.

All **10,047 committed tokens** reconcile with native token bytes, rendered
text, assigned condition, model calls and hash-chained journals. All **32 draw
transcripts** are distinct. All 90 package files match the existing installed
wheel and current SDK source; all 39 frozen engine files pass integrity checks.
The 27 relevant local long/fact-study tests pass. Review-builder changes preserve
the prior 96-output review page byte for byte. Private keys are not exported;
all public artifacts pass raw, hexadecimal and base64 key scans.

```sh
PYTHONPATH=src python tools/validate_long_fidelity.py \
  --model /path/to/545dc4251c05440727734bcd94334791f6ab0192 \
  --prior /private/multikey-prose-2026-09-24 \
  --output /new/private/long-fidelity
# Review public/blind-review.json before opening private/runs.json.
# Freeze per-fact, unsupported-addition, language and prose ratings plus hashes.
PYTHONPATH=src python tools/summarize_long_fidelity.py /new/private/long-fidelity
python tools/build_fact_review.py \
  /new/private/long-fidelity/public/blind-review.json review.html
```

The next experiment should establish a capable ordinary-generation baseline on
these factual tasks, using a stronger suitable local model with a pinned runtime,
before comparing its marked outputs. Preserve these failed attempts and the
original criteria. Do not weaken the watermark or repair generated responses to
make this study pass. No SDK default, scientific status, hosted CI, package
publication or public launch changes follow from this result.
