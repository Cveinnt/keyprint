# Where repeated answers come from

Real inference isolates strong concentration in the current 30-layer tournament.
At all **568 committed steps**, the active implementation's probabilities match
the separate reference implementation bit for bit. Ordinary probabilities remain
unchanged. This rules out a discrepancy in the optimized transform on these
inputs; it does not prove every part of the SDK correct.

Across the four marked answers, **92 of 100** positions whose base maximum token
probability was at most 99% ended above 99% after the transform. The other 212
positions already exceeded 99% in the base distribution. Altogether, 304/312
marked positions exceeded 99% after transformation.

This comparison uses the **same model head and prefix** before and after each
transform. It does not compare unrelated ordinary and marked answer positions.

| Existing key slot | Marked steps | Mean base entropy, bits | Mean marked entropy, bits | Previously non-concentrated positions pushed above 99% | Exact matches to three previous marked outputs |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 81 | 0.2070 | 0.0054 | 21/22 | 3 |
| 1 | 60 | 0.2184 | 0.0384 | 17/20 | 0 |
| 2 | 96 | 0.2894 | 0.0173 | 32/34 | 0 |
| 3 | 75 | 0.2492 | 0.0150 | 22/24 | 3 |

All four ordinary outputs differ from their corresponding previous outputs.
Two marked outputs also differ, so the sampler is **not strictly deterministic**.
The full recorded token path for key 0 has conditional probability about 0.917
under its prepared distributions, versus about 0.000038 under the base
distributions evaluated along that same path. This explains why independent
random draws can repeatedly produce the same answer. It is not an estimate of
repeat rates across other prompts or hardware/model executions.

## What this means for quality

Key 2 again describes air molecules reflecting light upward. Its wording changed
slightly but the scientific error remains. The other outputs and all intermediate
measurements are retained in [results.json](results.json). These outputs were not
blindly rated and do not add a new quality pass count. All eight are French; this
is still not a universal language or semantic-preservation guarantee.

The model already assigns mass to inaccurate continuations. Concentration can
make a chosen continuation recur under a fixed key. These measurements identify
that mechanism, not the counterfactual cause of every factual error. Sampling
variation, task fidelity and detection power must be evaluated separately.

## Method and reproducibility

- One unchanged French prompt; every existing September 24 key, including the
  failing key. One ordinary and one marked generation per key. Eight attempts,
  no retries, replacements, key selection or output rewriting.
- Pinned Qwen3-8B-4bit revision
  `545dc4251c05440727734bcd94334791f6ab0192`; MLX bounded reference; temperature
  0.7, top-k 100, cap 192. Every attempt ends at EOS, with exact token/render
  reconciliation, recorded model calls and distinct random-draw transcripts.
- [Plan](plan.json) saved before inference, binding script, case and prior-study
  hashes. [Summary](summary.json) binds the complete [measurements](results.json).
  [Checksums](sha256.json) cover the exported JSON artifacts. Keys and raw random
  draws remain private. Public records contain no key material.
- `tools/audit_probability_concentration.py` temporarily observes the actual
  session's `prepare` and `commit` calls. It returns the original prepared object,
  samples with the original RNG and never replaces the prepared probabilities.
  The reference implementation and a stage-by-stage replay independently check
  the final probabilities. Ordinary calls are checked against their input.
  Both implementations share the numeric tournament kernel; this is not an
  independent mathematical proof of that kernel. All observed base/prepared
  probability hashes and token IDs also match the SDK's own sampling receipts.
- All intermediate layers use the **original 30-layer PRF domain**. Their entropy
  traces are not evaluations of a separately configured lower-layer profile:
  changing layer configuration also changes that profile's domain and key bits.
- Added tests verify identical probabilities, random-draw transcripts, selected
  tokens and session receipts with/without the observer; repeated contexts,
  subnormal support, exception cleanup and incomplete evidence are covered.
  Those tests plus existing sampling tests: **37 passed**. Frozen engine manifest:
  **39 files pass**. The SDK and its default sampler are unchanged.

Run the audit with the existing private study and pinned local model:

```bash
PYTHONPATH=src python tools/audit_probability_concentration.py \
  --model /path/to/545dc4251c05440727734bcd94334791f6ab0192 \
  --prior /path/to/multikey-prose-2026-09-24 \
  --output /new/private/audit-directory
python tools/summarize_probability_concentration.py /new/private/audit-directory
```

The original private keys are required to reproduce these exact keyed profiles;
they are intentionally not distributed. New keys constitute a new experiment.
Observation adds significant work: its runtime cannot establish serving overhead.

## Release decision and next experiment

This resolves the immediate diagnostic question about the optimized transform;
it does not close A02, A03, detector calibration or launch readiness. Do not
advertise unchanged diversity for a fixed key or silently rotate keys to conceal
this result.

Next, evaluate a separately identified candidate that limits concentration,
retaining this profile as the reference. Freeze the candidate, workloads, keys,
quality criteria and detector evaluation before inference. Retain failures and
compare detection power at the same false-positive target alongside task fidelity
and repeated-prompt diversity. A weaker signal with prettier text is not a fix.
Changing layers or mixing distributions must not inherit the old detector's
acceptance. No default, release threshold or scientific status changes here.
