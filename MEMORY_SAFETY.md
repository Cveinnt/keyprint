# Bounded local research runs

The unguarded source-grounded replay was stopped after its macOS physical
footprint reached 22.8 GiB, with a recorded peak of 23.2 GiB. Most allocations
were attributed to IOAccelerator. This does not establish a ChatGPT application
leak or distinguish live MLX arrays from retained allocator buffers.

All 111 completed replay paths and 21,254 matched steps remain retained. A
separate guarded continuation subsequently verified the remaining 17 paths,
bringing the combined replay to 128 paths and 24,465 matched steps. Original
generated outputs, frozen factual ratings and interrupted attempt directories
remain unchanged. Do not restart the original unguarded runner.

## Guard policy

Future local MLX research runs must use both tools below. These are research
utilities, not automatic resource limits installed by the SDK.

```sh
PYTHONPATH=src:tools python tools/memory_watchdog.py \
  --output /absolute/path/to/new-attempt \
  --limit-gib 10 --timeout 1800 -- \
  python tools/mlx_guarded_worker.py /absolute/path/to/target.py [target arguments]
```

- One guarded research run per user, enforced with an exclusive lock. No automatic restart.
- Before launch: normal system memory pressure and at least 2 GiB free disk.
- Default 10 GiB process-group physical-footprint budget; smaller budgets supported.
  Measurements include compressed and Metal allocations, rather than relying on RSS.
  Shared pages can be counted more than once across processes.
- Every 250 ms, sample footprint, memory pressure and disk space. Stop on budget,
  pressure, disk or time failure, or a failed sensor. Process enumeration can add
  latency. This is a sampled termination threshold with possible overshoot,
  **not an OS allocation quota**.
- Send TERM, then KILL after 500 ms if necessary; check that the group has no
  live processes. Cleanup errors produce a failure receipt, never success.
- MLX retained buffer cache disabled. The 8 GiB MLX allocation setting is only
  an allocator guideline; the external watchdog supplies the cutoff.
- Log MLX active, cached and peak memory every 500 ms. The MLX worker checks for
  watchdog loss on that thread and exits when detected. This is a best-effort
  orphan check, not an OS guarantee if native code stalls the Python thread.
- Worker output and telemetry stream to files. Existing attempt directories are
  rejected. Keep `worker.log`, `memory.jsonl`, `mlx-memory.jsonl` and `result.json`.
  Missing final receipts after abrupt termination mean interrupted, not completed.

Run only a foreground target whose descendants stay in its process group.
Detached processes, unrelated apps and unwrapped SDK usage are outside this
guard. The lock prevents overlapping guarded research runs, not unrelated jobs.
Inspect global memory pressure before resuming work.

## Verified locally

Eighteen checks pass in the macOS MLX environment, without downloading or loading
a model: actual over-budget termination including TERM-resistant workers,
descendant cleanup after the leader exits, cancellation, deadline, sensor failure,
memory pressure, disk shortage, failure-receipt retention, concurrency exclusion,
real GPU allocation accounting, cache release and watchdog-loss recovery.

A separate 64 MiB Metal allocation probe completed under a 512 MiB budget with
180,388,440 bytes (~172 MiB) peak physical footprint. MLX active and cached bytes
both returned to zero. This qualifies the small probe, not sustained model usage.

```sh
PYTHONPATH=src:tools python -m pytest -q \
  tests/test_memory_watchdog.py tests/test_memory_mlx.py
```

The first guard tests exposed macOS EPERM on zombie-only process groups. Cleanup
now reaps the leader, excludes zombies, and tolerates EPERM only after confirming
no live group members. Genuine permission failures remain recorded failures.

The separately identified model preflight now passes: the first eight recorded
steps of the first original ordinary/marked pair match all sixteen native-logit
hashes, reference-weight hashes, original exact draws and tokens. Pinned local
Qwen3.5 reached 6.82 GiB peak physical footprint under the 10 GiB guard; cache
remained disabled and cleanup succeeded. This is prefix qualification, not a
full-path or sustained-run result.

A separately recorded continuation of the remaining seventeen paths was then
stopped by system memory pressure during initialization, at 5.09 GiB, before any
new path completed. TERM and cleanup succeeded; no automatic restart occurred.
Do not reinterpret that interruption as a sampling mismatch or a passed run.
All prior 111 completed paths remain unchanged. See the
[resource validation](evidence/memory-guard-2026-09-29/README.md).

Before another continuation, establish sufficient system headroom. Preserve the
interruption and changed resource policy; never overwrite an attempt directory.
`tools/continue_source_sampling.py` verifies the successful guarded preflight,
original script/study bindings, every completed prefix trace and the unchanged
SDK/model/runtime before continuing every remaining path in original order.
It requires the unchanged complete-cohort validator before producing a summary.

The separately namespaced paced candidate has now completed its own actual model
preflight: two eight-token prefixes and sixteen audited integer draws, under the
same watchdog and cache-disabled MLX wrapper. Sampled peak footprint was 6.48 GiB,
pressure remained normal and cleanup succeeded. Both prefixes hit their planned
eight-token cap; this is not full-output quality or sustained serving evidence.
A separate 128-attempt paced study is subject to the same resource policy. See
[candidate preflight](evidence/paced-native-preflight-2026-09-29/README.md).

## Completed guarded runs and recovery checkpoint

The full paced study completed 128 EOS outputs and 24,421 audited committed
tokens. Peak sampled footprint was 6.90 GiB over 29.8 minutes; all 6,304 pressure
samples were normal, cached bytes stayed zero, and cleanup succeeded. See
[complete candidate run](evidence/paced-native-run-2026-09-29/README.md).

After that run established headroom, the separately recorded original replay
continuation completed under the same guard: 6.65 GiB peak over 232.57 seconds,
824 normal-pressure samples, zero cached bytes and verified cleanup. Earlier
memory and pressure interruptions remain retained; this was not an automatic
retry. Completion establishes mechanical replay, not factual quality.

The task-local `../../receipts/source-grounded-next-2026-09-28/CONTINUE.md`
checkpoint records terminal jobs, evidence paths, review progress and next steps.
Update it before stopping work or restarting the app. Resume from verified
receipts; never infer success from a stale LIVE entry or restart a completed job.
The latest resource recheck is stored at
`../../receipts/memory-guard-2026-09-29/memory-safeguards-recheck-2026-09-29.json`.

All 18 guard tests passed again after these runs. No model loading was required
for that test suite. These safeguards apply to wrapped research workers; they
do not establish that a ChatGPT application memory leak has been fixed.

## Reduced budget for the balanced study

After the user authorized memory cleanup, the Android emulator and a Gradle
daemon verified IDLE were stopped gracefully; its Kotlin child also exited.
An attempt after stopping only the emulator still stopped on system pressure
before generating any token. Both interruption receipts remain retained.

A separately recorded balanced native preflight then completed all four
16-token prefixes with 64 audited draws. It peaked at 5.52 GiB under a reduced
**8 GiB** sampled cutoff, with disabled MLX cache, normal pressure throughout
and verified cleanup. The full frozen balanced study uses the same reduced
cutoff. See [preflight evidence](evidence/balanced-native-preflight-2026-09-29/README.md).
The guard still stops on pressure, disk, deadline or sensor failure. No automatic
restart, unrelated-app cleanup, quality acceptance or ChatGPT leak fix is implied.
