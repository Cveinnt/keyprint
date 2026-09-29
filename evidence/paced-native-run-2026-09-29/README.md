# Complete paced-candidate native generation

The fixed candidate study completed **128/128 outputs**, all at EOS, with
**24,421 committed tokens**. Every output passed its in-run sampling/render audit;
the separate complete-cohort reconciliation passed. No generation, decoding or
sampling-audit errors occurred, and no output reached the 768-token cap.

This completes actual generation and receipt reconciliation. All 128 assistant
source-only ratings have since been frozen and summarized in the
[complete quality review](../paced-quality-2026-09-29/README.md). Strict full-task
counts are 25/64 ordinary versus 16/64 marked; accepting all preflagged uncertainty
gives 36/64 versus 34/64. Quality and detection remain unqualified.

## Scope and invariants

The study uses the original sixteen source contexts, four existing keys, all
ordinary/marked combinations and the same ordering, temperature 0.7, top-k 100 and
768-token limit. The new paced profile and fresh recorded OS sampling randomness
are separately identified. Source text, derived outputs, keys and detailed
review material remain private, separate from the MIT SDK.

Receipts retain native-head hashes, sparse base weights, integer distributions,
random-bit transcripts and token commits. The in-run auditor replays the sampling
session, checks rational CDF selection and exact excluded/global probability
bounds, and reconciles rendered text with committed UTF-8 bytes. This shares the
research kernel and does not independently replay native model heads.

The cohort auditor verifies every scheduled attempt, final result, output/journal
binding, original source/rubric content and blinded review record. It cannot turn
a partial cohort into success. No response retry, text repair or replacement
output was used. All 78 frozen SDK Python sources and planned research helpers
were rechecked unchanged after generation.

## Resource outcome

The 10 GiB watchdog completed normally after 1,785.48 seconds, approximately
29.8 minutes. Sampled peak physical footprint was **7,411,962,536 bytes
(6.90 GiB)**. All 6,304 recorded pressure samples were normal. MLX cache remained
at zero; final telemetry recorded ten active bytes and zero cached bytes.
Process-group cleanup was verified.

These figures include generation, trace writing and sampling audits on a shared
machine. They are not an isolated serving-overhead comparison. The watchdog is a
sampled termination threshold with possible overshoot, not an OS allocation quota.
This run does not prove or diagnose the reported ChatGPT application memory leak.

## Review and remaining work

A private interactive review page contains all 128 source/output pairs, preserves
every row, hides assignment metadata and starts with blank human ratings.
Assistant ratings are complete and remain separate from human acceptance.
The [review protocol](../paced-quality-protocol-2026-09-29/README.md) required all
ratings and uncertainties to be frozen before the condition/signal join.

Next: independent quality and detection qualification on prospectively frozen,
disjoint material, then serving qualification. The [original replay](../source-replay-complete-2026-09-29/README.md)
now also verifies all 128 paths; candidate generation does not replace it.
No SDK promotion, public deployment, 25/25 conformance or launch acceptance follows.

Recorded [plan](plan.json), [profile identity](identity.json),
[final generation result](results.json) and [resource/audit validation](validation.json).
