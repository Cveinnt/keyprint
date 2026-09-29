# Actual paced-candidate model preflight

The guarded native candidate preflight completed on the pinned local Qwen3.5
model: **two eight-token outputs, sixteen audited committed draws**, no execution,
decoding or sampling-audit errors. Both outputs ended at the planned eight-token
cap. These deliberately short prefixes do not establish output quality, normal
EOS behavior, detection performance or sustained serving cost.

The first original ordinary/marked pair was fixed before execution. Source
prompts, the original key, frozen SDK, model assets and runtime were verified.
Fresh OS randomness was recorded; this generates new prefixes, not a replay of
the old outputs. Receipts retain native-head hashes; the sampling audit reconciles
integer draw receipts, token commits and exact output bytes. It does not
independently rerun native model heads. All source-derived text remains in private
receipts.

## Resource receipt

The external watchdog completed normally in 17.66 seconds, with a sampled peak
physical footprint of **6,960,698,096 bytes (6.48 GiB)** under the 10 GiB cutoff.
System pressure stayed normal throughout the recorded samples. MLX cache remained
disabled; the final telemetry recorded ten active bytes and zero cached bytes.
Process-group cleanup was verified. A sampled cutoff can overshoot and is not an
OS allocation quota. This does not explain or prove a ChatGPT application leak.

A separately identified full 128-attempt study has been started under the same
memory policy. Its progress must not be treated as a completed cohort. Old
interrupted replay evidence remains retained separately at 111/128.

## Complete-cohort and review preparation

`tools/audit_paced_study.py` requires the final 128-attempt result, original
schedule, frozen source/rubric content, every output/journal binding and a complete
blinded review packet. Errors and caps remain in the denominator. It reconciles
the in-run sampler audits; it does not independently replay native inference or
make factual judgments. Equivalent JSON Unicode escaping is permitted only when
parsed source content matches the hash-verified original files exactly.

`tools/build_paced_review.py` adapts that complete verified packet to the existing
private source-review UI. Conditions, keys and signal counts stay hidden. All
ratings start blank. Decoder failures remain visible execution errors; sampling
audit errors block preparation. No human acceptance is inferred.

Fifteen new local tests pass for these tools: complete and failed cohorts,
immutable originals, missing final receipts, omitted/duplicated attempts,
changed assignment/counts/journals, review metadata leakage, safe source escaping,
and preservation of every review record. These tests use synthetic receipt
fixtures; they are additional to the prior 141 focused checks, not a fresh
full-SDK suite.

After the live study completes, use fresh output artifacts:

```sh
PYTHONPATH=src:tools python tools/audit_paced_study.py /private/paced-study \
  --original /private/original-source-run
PYTHONPATH=src:tools python tools/build_paced_review.py /private/paced-study \
  /private/new-review --original /private/original-source-run
```

Recorded [plan](plan.json), [profile identity](identity.json),
[preflight result](results.json) and [source/resource validation](validation.json).
SDK defaults, production website, publication and launch acceptance are unchanged.
