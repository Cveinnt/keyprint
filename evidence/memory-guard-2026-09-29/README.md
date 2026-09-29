# Resource validation and retained replay interruption

This closes a local research-process safety deficiency, not a watermark-quality
or launch criterion. The SDK, sampler, original generated text and factual
ratings are unchanged. Public release remains held.

| Attempt | Scope | Result |
| --- | --- | --- |
| Original diagnostic replay | All 128 original paths planned | Stopped at 22.8 GiB footprint; 111 completed paths and 21,254 matched steps retained. Interrupted path not counted complete. |
| Small Metal probe | One 64 MiB allocation under a 512 MiB budget | Completed; ~172 MiB peak physical footprint; active/cache bytes returned to zero. |
| Guarded model preflight | Fixed first pair, first eight recorded steps each | 16/16 native heads, reference weights, original draws and tokens matched. 6.82 GiB peak; normal memory pressure; cleanup verified. |
| Guarded continuation | All remaining seventeen original paths, in order | System-pressure stop during initialization at 5.09 GiB; zero new completed paths; TERM and cleanup verified. |

The continuation records its own plan and preserves the original partial run.
It does not manufacture a final original-run receipt or overwrite any generated
output. The interrupted path would restart from its original prompt with its
original draws, explicitly recorded as a continuation after a resource failure.
The two eight-step preflight prefixes do not add to the original 111 completed
paths or establish full replay, detection power, factual fidelity or launch readiness.

Eighteen prior watchdog/Metal checks passed. The subsequent 37 focused
preflight/continuation/summary checks cover fixed selection, rejected partial or
changed prefix evidence, source/study/guard bindings, no weakened native replay
loop, and complete-cohort summarization. These are separate scoped checks, not a
fresh full-SDK suite. The real model preflight is separate inference evidence.

No further model run was automatically restarted after the pressure stop.
System pressure returned to normal after cleanup. A contemporaneous process
snapshot showed substantial unrelated VM/emulator use; those processes were not
stopped. This does not attribute all system memory pressure to a single app or
prove a ChatGPT memory leak.

See [guard policy](../../MEMORY_SAFETY.md) and [aggregate receipt](validation.json).
Raw logs and original source-derived data remain in the private study receipts.
