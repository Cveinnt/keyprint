# Balanced native preflight completed within the reduced memory budget

Four native Qwen3.5-9B prefixes completed: English and Spanish, ordinary and
marked, 16 tokens each. All 64 committed tokens passed exact in-run sampler
checks. All four hit the planned short cap; none is a full-output quality result.
No execution, decode or sampling audit failures occurred. The completed receipt
passes `validate_preflight`, including helper/profile/SDK/model/runtime bindings.

The worker peaked at 5,931,569,952 bytes (5.52 GiB), below the newly reduced
8 GiB sampled cutoff. It finished in 11.44 seconds, all 40 pressure samples were
normal, the MLX retained cache stayed zero and cleanup was verified. This is a
sampled cutoff with possible overshoot, not a hard OS allocation quota.

Memory was freed under the user's explicit cleanup authorization. The Android
emulator was stopped gracefully. A first subsequent preflight still encountered
system pressure at 5.26 GiB before generating tokens; its failed attempt remains
retained and is included here. Gradle then confirmed its 3.93 GiB daemon was IDLE;
graceful shutdown also exited its 1.67 GiB Kotlin child. All three processes were
confirmed gone. Other apps and virtual machines were left running. After six
normal pressure readings, a separately recorded preflight succeeded. No automatic
restart or overwrite occurred, and the pressure criterion was not relaxed.

This closes native execution preflight only. It does not establish language or
meaning preservation, detection power, calibration, sustained serving, other
backends or launch readiness. The frozen full study uses the same profile and
resource guards. Research-only code has not replaced the public SDK defaults.
