# Bounded batches: tested recovery, actual-study validation pending

The research runner can now preserve multiple interrupted ancestors and stop at
a declared boundary between complete outputs. Each invocation has a fixed
maximum number of new attempts, defaulting to four. It unloads the worker after
that batch; it does not start the next batch automatically. This changes process
lifetime, not source inputs, sampling settings, keys, token caps or review rules.

The chain verifier checks the original frozen study and first continuation,
then every additional ancestor's checkpoint, guard receipt, file inventory,
schedule prefix, retained results, journal hashes and helper bindings. All prior
results are copied unchanged. A journal interrupted mid-output becomes an
explicit unavailable infrastructure failure; no text is regenerated. A verified
pause between outputs adds no failure. Historical pause receipts remain bound.

The runner requires the existing external watchdog and cache-disabled wrapper.
SDK, runtime, successful preflight, model, tokenizer, prompt IDs and profile
bindings remain checked. Finalization requires all 128 outcomes, including every
failure. The chain audit supplements the original complete-cohort audit, which
does not independently validate ancestry. Source-only judgments must still be
frozen before any condition, key or score join.

## Verified locally

Twenty-one new synthetic tests plus existing continuation, study and inference
tests passed: **83 checks in 8.03 seconds**. They exercise preservation of two
interruptions, repeated planned batches, changed ancestor/child rejection,
retention of unavailable scores, output-count limits, no plan-only execution,
an interruption in the final scheduled attempt and complete-cohort accounting.
These tests load no model and do not establish native inference or quality.

An actual saved-study plan-only check was attempted under a **512 MiB** CPU
process budget. The watchdog refused launch because system memory pressure was
already elevated. It recorded zero samples and no worker PID; the requested
output directory was never created. Four subsequent readings also reported
warning pressure. The failure receipt is retained; no automatic restart or
unguarded substitute was used. Existing study data remains unchanged.

Therefore, actual-study validation of this new batch harness is **pending**.
There are still 71 complete outputs, two retained interruptions and 55 unstarted
attempts. No additional quality, detection, compatibility or launch requirement
has closed. Public release, registry publication and deployment remain held.

## Next execution

When system headroom is normal, first run the plan-only CLI against the retained
ancestry and verify its lineage without loading a model. Only then consider one
explicit four-output native batch under the existing 8 GiB pressure/disk/time
guard. Preserve its terminal receipt and check memory before any subsequent
invocation. Do not change bound helpers after an actual native batch begins.
