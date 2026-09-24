# Four-key prose follow-up, September 24

All 72 actual generations are retained. Marked outputs passed 32/36 fixed task
rubrics; ordinary outputs passed 31/36. All 72 stayed in the requested language,
completed, and reconciled exact rendered text with token traces.

| Task rubric | Ordinary | Marked |
| --- | ---: | ---: |
| Picnic without concrete invented date/time/place/name | 11/12 | 11/12 |
| Approval email preserving actors, deadline and prerequisite | 12/12 | 12/12 |
| Two French sentences with an adequate scattering explanation | 8/12 | 9/12 |
| Overall | 31/36 | 32/36 |

Paired outcomes: 28 both pass, four marked-only passes, three ordinary-only
passes, one both fail. Observed marked-minus-ordinary failure rate: -2.78
percentage points. No inferential harm bound is claimed for this small batch.

## Repetition persists across keys

Each key produced the same marked French answer in its three repetitions.
Answers differed across keys. Three keys passed the French rubric 3/3; one key
failed 0/3 with an incorrect upward-reflection explanation. Ordinary French
answers were distinct in every within-key block, passing 2/3 in each.

| Key slot | Marked French passes | Distinct ordinary French outputs | Distinct marked French outputs |
| --- | ---: | ---: | ---: |
| 0 | 3/3 | 3/3 | 1/3 |
| 1 | 3/3 | 3/3 | 1/3 |
| 2 | 0/3 | 3/3 | 1/3 |
| 3 | 3/3 | 3/3 | 1/3 |

The earlier one-key quality result did not generalize to every new key.
Repetition did. All 72 attempts recorded model calls and distinct random-draw
transcripts; committed token counts match. This rules out simply reusing cached
responses in this batch, not all implementation issues or statistical RNG defects.
Do not advertise unchanged output diversity under a fixed key. Key-conditioned
sampling is a plausible explanation, but this experiment does not isolate each
sampling stage. The engine was not changed or tuned.

## Method and limits

Prompts and rubrics were unchanged from September 23. Four independent fresh
keys, three repetitions, three prompts, both conditions; no retries, favorable
key selection, replacements, rewriting or grammar constraints. Model:
Qwen3-8B-4bit revision `545dc4251c05440727734bcd94334791f6ab0192`, MLX reference,
temperature 0.7, top-k 100, cap 192 tokens.

The assistant reviewed condition/key-hidden outputs, grouping identical strings
to apply ratings consistently. All 72 ratings were frozen before revealing
mappings. This is not independent human review. The narrow picnic rubric accepts
some unsupported weather/readiness elaborations because it tests concrete
date/time/place/name additions; it is not general factual or prose approval.
One ordinary and one marked picnic invented a park and noon. French failures
concern completeness or scientific content, not translation. Ambiguous phrasing
is noted in ratings. Reused prompts are not independent topic-level trials.

## Inspect or reproduce

- [Plan and unchanged rubrics](plan.json)
- [Condition-hidden outputs](blind-review.json)
- [Frozen ratings](frozen-ratings.json) and [commitment](rating-commitment.json)
- [All outputs, per-key breakdown and freshness audit](results.json)

`tools/multikey_prose_pilot.py` creates a new batch with the exact pinned snapshot.
After completing hidden review, `tools/summarize_multikey_prose.py` verifies
commitments and joins the retained private mapping. Public receipts exclude keys
and raw draw journals. `sha256.json` covers the five exported JSON receipts.

This advances comparative evidence and identifies a repeatability limitation.
It does not close A02, A03, detector calibration or the full launch goal.
