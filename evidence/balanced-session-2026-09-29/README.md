# Balanced research integration: local checks pass; native preflight interrupted

The exact balanced allocation rule now has its own experimental profile/domain,
session, native inference loop and token-path score replay. Old paced profiles
are rejected by the new session, and the old session rejects the balanced
profile. SDK sources and defaults remain unchanged; this is not a public API
or inherited detector calibration.

The new session reuses the existing exact integer draw/commit lifecycle and
explicit source-alignment decisions. It overrides preparation, numeric policy
and receipts. Protected source continuations, formatting groups and EOS retain
their normalized probabilities exactly. Repeated contexts and single-label
supports use identity. Eligible probabilities stay in [p/2,3p/2]. No completed
text is translated, rewritten or silently repaired. Those mechanical properties
do not establish that sampled text retains meaning or stays in its language.

The inference loop saves every prepared integer distribution, draw transcript,
committed token and failure. It checks the decoder against exact committed UTF-8
bytes and rejects changed text. The audit replays sampler state and independently
checks rational CDF intervals and protected mass. It does not independently
recompute model heads. Token-path score replay requires the explicit new profile
and rejects tokens after EOS; arbitrary-text replay and calibration are still
unqualified.

## Current evidence

- 38 new session/inference checks pass: profile isolation, exclusions, grouped
  ratios, overlapping source matches, repeated contexts, EOS, owner/step limits,
  foreign prepared states, retained rejection failures, smallest positive
  binary64 support, exact bytes, incomplete forward attempts and tampering.
- 135 combined checks pass across new kernels, sessions/inference and the old
  paced session/inference regression suites. No model was loaded for these tests.
- A four-attempt native preflight was planned before loading the pinned local
  Qwen3.5 model: fresh keys, English and Spanish prompts, ordinary/marked pairs,
  sixteen-token caps and no retries. This is prefix execution testing, not a
  quality or multilingual acceptance study.
- The watchdog stopped the run when global memory pressure reached warning (2),
  at 5,720,970,016 bytes (5.33 GiB) sampled peak. It stopped before any token
  committed: one retained journal contains only its start event. No final run
  or preflight result exists. Model/binding identity was created, which does not
  establish successful inference.
- All observed MLX cache samples were zero. Cleanup was verified; no automatic
  restart occurred. The 10 GiB threshold was not reached: global pressure was
  the stopping condition. This does not diagnose ChatGPT's reported memory use.

The interruption, fresh keys and partial journal remain in private receipts.
Plan/source hashes and unchanged SDK source hashes were rechecked. Do not
overwrite the attempt, count missing outputs as passes, or resume it merely
because a later pressure snapshot is normal. Establish resource headroom before
an explicitly recorded new attempt in a fresh directory.

## Next qualification

Finish a guarded native preflight with the same declared mechanics before a
full study. Prepare disjoint synthetic source passages and fact rubrics covering
English, Spanish, French and Chinese, including names, numbers, negation and
conditional requirements. Freeze those inputs, fresh keys, both conditions,
generation caps, failure handling and review criteria before generating outputs.
Keep prompt/source language aligned; no translation step is part of watermarking.

Retain every output and failure, score source-only quality before revealing
conditions or detector counts, and keep any human review distinct from assistant
judgments. Evaluate raw-text/token-path replay separately and test detection
against fresh null controls; a handful of successful examples cannot qualify
false-positive rates or universal semantic preservation. This protocol direction
is not yet a frozen full-study manifest. Existing failures and the launch hold
remain in force.

[Validation](validation.json), [interrupted plan](interrupted-preflight-plan.json),
[loaded profile identity](profile-identity.json), [resource stop](resource-stop.json).
