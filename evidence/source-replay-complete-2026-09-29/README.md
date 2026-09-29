# Complete original native sampling replay

All 128 original generated paths now verify across 24,465 steps. The original 111 completed paths remain unchanged; a separate guarded continuation verified the remaining 17. Original interrupted attempt directories are retained. No new generation, random draws or repaired outputs were introduced.

All 128 trace hashes were independently rechecked, and the unchanged complete-cohort summarizer was run again against original records and retained/new traces. Its recomputed result exactly matches the saved summary. This closes the unfinished mechanical replay obligation, not factual quality or semantic causation.

Within the marked paths, mean conditional entropy falls from 0.4066 to 0.04893 bits, mean maximum token probability rises from 0.8993 to 0.9867, and mean total variation is 0.1309. There are 4,204 steps newly above 99% maximum probability. Maximum absolute EOS-mass change is 2.78e-17. Ordinary distributions are unchanged. Measurements share the recorded prefix within each step; ordinary and marked generation paths differ, so these measurements do not isolate the cause of factual errors.

The second continuation completed under the 10 GiB watchdog at 6.65 GiB sampled peak, in 232.57 seconds. All 824 memory-pressure samples were normal, cached bytes stayed zero and process cleanup succeeded. The first pressure-stopped continuation and original unguarded memory stop remain preserved. This was a separately justified resource attempt, not automatic restart.

[Summary](summary.json) binds all traces and the complete result. [Validation](validation.json) records verification and resource scope. These are original-profile diagnostics and do not qualify the separately named paced candidate or establish detector calibration.
