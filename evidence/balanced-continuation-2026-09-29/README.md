# Balanced continuation: preserved evidence, incomplete study

The continuation preserves all 63 original completed outputs byte-for-byte and
records the original interrupted attempt as an explicit infrastructure failure.
Its 19 committed journal tokens are retained, with no completed text or detection
score fabricated. Only previously unstarted schedule entries may generate.
The original study directory remains unchanged.

Seventeen new continuation tests and existing study/inference checks passed:
62 tests in 6.51 seconds. They cover altered ancestry, prompts, journals, output
order, helper bindings, dropped failures and premature finalization. A plan-only
execution against the actual retained study passed without loading a model.
The additional continuation lineage audit is required alongside the ordinary
full-cohort auditor; the latter alone does not validate ancestry. Chaining a
second interrupted continuation is not supported by this harness.

After graceful shutdown of a verified IDLE Gradle daemon and its Kotlin child,
seven successive pressure samples were normal. One guarded continuation then
completed eight additional EOS outputs, totaling 1,009 committed tokens, before
system pressure triggered termination. Peak sampled worker-group footprint was
6,205,870,904 bytes (5.78 GiB), below the 8 GiB cutoff. Cleanup was verified.
No automatic restart followed. These safeguards do not diagnose or fix memory
growth in the ChatGPT app or unrelated workloads.

Current accounting across both attempts:

| Outcome | Count |
| --- | ---: |
| Complete EOS outputs | 71 |
| Completed-output committed tokens | 10,096 |
| Original sealed infrastructure interruption | 1 |
| New unfinished attempt, 107 journal commits retained | 1 |
| Unstarted attempts | 55 |
| Total scheduled attempts | 128 |

All eight new completed rows reconcile their result and journal hashes and
retained in-run sampler audits, with zero execution, decode or audit errors.
This is receipt reconciliation, not independent native model-head replay or
semantic acceptance. Both unfinished attempts remain failures to complete.

Source-only assistant ratings now cover 71 completed texts plus the original
sealed failure. Prior ratings remain unchanged; uncertainties were recorded
before joining any conditions, keys or scores. One newly reviewed output omits
required sensor warnings. This observation cannot yet be attributed to the
watermark condition. The all-128 rating freeze and complete quality/detection
analysis remain unavailable. No partial quality score is promoted as a result.

SDK defaults, production design and publication hold remain unchanged. No public
launch, registry publication, deployment or CI activation occurred. The next
model execution needs sustained resource headroom and a separately validated
continuation preserving both interruptions and every completed output.
