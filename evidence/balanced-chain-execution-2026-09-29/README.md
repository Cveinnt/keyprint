# Actual ancestry validation passed; native batch stopped before first token

The saved-study plan-only check completed under a 512 MiB process budget, peaking
at 82,248,256 bytes (78.44 MiB). All seven pressure samples were normal and cleanup
was verified. It validated both ancestors, preserved 72 recorded outcomes, and
sealed the second interrupted attempt as an explicit failure. No model was
loaded by this plan-only invocation.

With pressure normal and more than 6 GiB free disk, one separately guarded
four-output native batch was started. It reached the first attempt's journal
header, but pressure stopped it before any prepared/drawn/committed token event.
Peak sampled footprint was 5,844,374,400 bytes (5.44 GiB); elapsed time was 11.27
seconds. Cleanup was verified. No new output or final study result exists.

Current accounting is **71 complete outputs, three interrupted attempts and 54
unstarted attempts**. The third interruption contains a start header and zero
commits. It remains retained, without retry or a fabricated detection score. The
two earlier interruptions are sealed in the copied study; original artifacts
remain unchanged. All 73 recorded outcomes have source-only assistant ratings,
including unavailable failed completions. Conditions, keys and scores have not
been joined to those ratings.

The user declined stopping Docker Desktop and reiterated low memory use and
sequential execution. Docker remains running. Keyprint already runs one guarded
worker at a time; sequential batches do not eliminate the observed 5.4 GiB
startup footprint. Further local model starts are held under this constraint.
Do not stop other applications, relax pressure protection, change the frozen
model or repeat attempts based solely on transient normal-pressure readings.

Only our hung read-only Docker inventory command was terminated. Docker's VM,
daemons and another task's Docker command remained untouched. Process snapshots
are diagnostics, not evidence that an application leak was fixed.

The prior 83 synthetic/regression tests still apply; research code did not change.
Actual ancestry validation is now achieved. Native batch completion, full-cohort
semantic review, detection qualification and launch readiness are not achieved.
SDK defaults, approved site, CI-off policy and publication hold remain unchanged.

A separate lightweight SDK UI fix now aborts the outstanding progress-only
request when its watcher stops, as well as clearing its timer. Sixteen local
playground UI tests pass, including twenty repeated start/stop cycles and a
check that generation remains untouched. This prevents a stale status request
from remaining pending after cleanup. It does not diagnose the reported ChatGPT
memory growth or establish native inference quality. No site was deployed.
