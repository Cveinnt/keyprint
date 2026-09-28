# Half-mixture development study

This experiment tests one fixed candidate: average the ordinary token
distribution and the current 30-layer marked distribution before sampling.
It does not rewrite generated text, select favorable keys or rotate keys between
requests. The installed SDK and its default sampler are unchanged.

## Result: reject this candidate

All 48 corrected-run attempts completed at EOS without execution/scoring errors.
All 5,169 committed tokens reconcile with journals; all 48 random-draw transcripts
are distinct. Every output remains in its requested language.

| Measure | Ordinary | Current reference | Half mixture |
| --- | ---: | ---: | ---: |
| Fixed rubric passes | 13/16 | 13/16 | 12/16 |
| French explanation passes | 6/8 | 6/8 | 6/8 |
| Approval email passes | 4/4 | 4/4 | 4/4 |
| Longer English passes | 2/2 | 2/2 | 2/2 |
| Longer Spanish passes | 1/2 | 1/2 | 0/2 |
| Longer matching-key detections | 0/4 | 3/4 | 0/4 |
| Longer other-key detections | 0/4 | 0/4 | 0/4 |

Both mixture French responses differ within every key. The reference repeats
exactly in three of four key blocks; ordinary responses differ in all four.
This is eight French outputs per arm, not a general diversity estimate. Greater
variation did not improve aggregate task passes and yielded no hits across the
four longer outputs at the frozen rule. The failed key still produces factual errors: one
mixture answer incorrectly says air absorbs red/yellow light. It is not fixed by
replacing one recurring bad answer with a different bad answer.

The paired reference/mixture task outcomes are 11 both pass, two reference-only
passes, one mixture-only pass and two both fail. Both mixture Spanish outputs
exceed 200 words. Two borderline reflection/scattering phrases were conservatively
failed before revealing their arms: one ordinary, one mixture. Counting both as
passes would give 14/16 ordinary, 13/16 reference and 13/16 mixture. That does not
change the rejection on detection. No frozen rating is changed for that sensitivity
observation.

All twelve short reference outputs miss the same detection rule; that limitation
also remains open. Zero observed ordinary/other-key hits here cannot establish a
false-positive guarantee. This development study establishes neither quality
noninferiority nor detector calibration and closes no scientific acceptance.

Inspect [all long-form samples](REVIEW_SAMPLES.md),
[frozen ratings](frozen-ratings.json), [review commitment](rating-commitment.json)
and [complete results, per-key counts and receipt audit](results.json).
[Plan](plan.json), [execution identities](identities.json) and
[checksums](sha256.json) bind the retained evidence.

Validation: 52 candidate, caller, summary, concentration and sampling tests pass.
The frozen SDK engine manifest passes for all 39 files. No SDK source, package
dependencies, production website or hosted CI settings changed.

## Registered comparison

Four existing keys, including the known failing key. Per key: two repetitions of
the French sky prompt, one approval email and one longer explanation. Even key
slots receive an English weather/climate prompt; odd slots receive a Spanish
library prompt. Every task instance has ordinary, reference-marked and
half-mixture arms: 48 planned attempts. Their order rotates deterministically.
The existing French and approval rubrics are unchanged. Long answers have explicit
language, paragraph, 150–200-word and factual requirements.

Model: pinned Qwen3-8B-4bit MLX, revision
`545dc4251c05440727734bcd94334791f6ab0192`, temperature 0.7, top-k 100.
Short tasks cap at 192 tokens; longer tasks cap at 768. Caps and errors remain
in the record. No retries, output substitutions or threshold search.

The detector is the existing weighted-layer reference-tail rule, frozen before
generation. Matching and next-slot keys each use an inclusive 0.005 threshold,
corresponding to a **nominal** two-key family target of 0.01. Its previous
fixed-key null qualification failed. This study therefore compares signal at
the same rule; it does **not** establish equal actual false-positive rates or
1% deployment calibration. Longer outputs are the primary signal screen;
short-answer scores remain visible separately.

Promotion is rejected if the candidate has fewer rubric passes or fewer long
matching-key hits than the reference, or incomplete evidence. Equal or better
counts only justify a larger fresh study; these small samples cannot prove
noninferiority. SDK promotion and scientific acceptance remain false regardless
of this development-screen outcome.

## Execution and prior failure

The first attempt incorrectly tried to pass a new research profile through the
SDK's registered execution/reporting facade. Seven mixture attempts were rejected
with empty inference journals. Stopping the study interrupted a reference attempt;
another reference attempt remained unfinalized. The original 21 recorded attempts,
eight errors, plan and source snapshots are retained under
[aborted-sdk-run](aborted-sdk-run/abort.json). A separate
[erratum](aborted-sdk-run/abort-erratum.json) corrects the initial abort note's
ordinary/reference label. No failed attempt is silently replaced or treated as a
negative detector result.

The corrected run uses a dedicated research caller, separately identified by
candidate and caller source hashes. It uses the same filtered sampling pipeline,
durable draw journals and model prefill pattern, with its own explicit research
report. The SDK's execution allowlist is intact. Deterministic tests compare the
research reference runner's ordinary and marked draws, tokens and receipts with
the SDK; additional tests cover mixture arithmetic, subnormal support, excluded
mass, repeated contexts, EOS and unsupported channels.

Ratings hide condition/key metadata until commitment. They are assistant reviews,
not independent human acceptance; the rater has prior project context and may
recognize previously seen repeated text. Private keys and raw draw journals are
excluded from public artifacts. The script hashes bind the registered run.

The candidate remains outside `src/` and the installable package. Reference PRF
bits are retained for controlled comparison; the sampling law and research
execution identity differ. No detector acceptance transfers.

Next work should target the remaining detection/quality bottlenecks while
preserving the current reference. Do not adopt blanket signal dilution for a
more varied demo. Any different candidate or detector requires its own frozen
comparison and fresh evaluation; this failed candidate supplies no acceptance.
