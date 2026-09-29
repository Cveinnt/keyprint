# Paced research session: exact excluded mass and recorded draws

The paced kernel now has a separate research session combining source protection,
explicit EOS handling, repeated-context behavior and exact integer sampling.
This is synthetic lifecycle validation, not model inference, quality acceptance,
detector calibration or SDK promotion. All 78 SDK Python sources still match the
original source-grounded inference plan.

## Probability and lifecycle contract

`PacedProfile` binds the new policy and explicit EOS IDs into its immutable
profile identity. Old profiles are rejected; old calibration cannot transfer.
Source alignment reuses the frozen source policy: all overlapping canonical
suffix matches, a 24-byte match and 96-byte startup window. Only ASCII whitespace
is canonicalized; Unicode byte fragments are preserved.

For original integer weights `(b, B)`, active mass `M`, and transformed active
conditional weights `(r, R)`, global weights are `M*r` for active candidates and
`b*R` for excluded candidates, with total `B*R`. Common-factor reduction preserves
the distribution exactly. This retains normalized probabilities for EOS,
formatting and protected source continuations, including when floating-point
weights sum only approximately to one. Exact cross-product checks retain support
and whole-distribution half-to-double bounds.

The new prepared object intentionally has no floating-point probability field.
**Never convert its integer weights back to floats for sampling.** Doing so
invalidates the exact excluded-mass guarantee. Draws use explicit random bits and
bounded integer rejection sampling; all valid attempts are retained. Only the
recorded token can be committed, once. EOS closes the session. Numerical or
sampling failures close it without substituting another distribution.

The active kernel still admits at most 4,096 positive weights and minimum
normalized active base probability `2^-900`; unsupported input fails. Sparse
vocabulary coverage does not establish unrestricted full-head compatibility.
Protected continuation probabilities are preserved, not guaranteed to be selected.
None of these constraints proves preservation of every fact or meaning.

## Retained results

All **16 synthetic sessions passed**, with **171 committed draws**, 15 EOS endings
and one retained step-cap ending. They cover four public test keys, ordinary and
marked conditions, and general/proofreading purposes. Across these sessions,
23 integer partitions and 33 protected candidates were recorded. Every draw was
checked against an independently calculated rational CDF; raw replay events were
reconciled using the existing shared replay primitives. Fixtures use supplied
probabilities, not model logits, and test-only seeded randomness.

The first synthetic attempt also completed. Review then found that the added EOS
attribute needed explicit frozen-dataclass protection. That was fixed and tested;
this fresh, hash-bound attempt retains exactly the same sixteen trace hashes.
Both attempt directories remain in the private receipts. No failure was dropped.

**103 focused tests passed**, including 29 new session tests and the previous 74
numerical tests. Coverage includes excluded/EOS mass, overlapping source matches,
multiple keys, repeated contexts, immutable profiles/prepared objects, single-use
draw/commit behavior, invalid randomness, rejection failure receipts, thread
ownership, startup expiry, Unicode fragments, and sparse 248,320-token heads.
This is not a fresh full-SDK suite.

The separate 256 MiB watchdog recorded 67,584,528 bytes (~64.5 MiB) sampled peak
footprint and verified cleanup. No language model or GPU was loaded. Memory
thresholds are sampled termination limits, not OS allocation quotas.

## Reproduce and continue

```sh
PYTHONPATH=src:tools python -m pytest -q tests/test_paced_source_session.py
PYTHONPATH=src:tools python tools/memory_watchdog.py \
  --output /path/to/new-supervisor --limit-gib 0.25 --timeout 120 -- \
  python tools/validate_paced_session.py --output /path/to/new-study
```

Next: an explicitly identified actual-inference harness consuming this integer
contract, followed by retained paired outputs, factual/language review, independent
ordinary/wrong-key calibration and serving measurements. Heavy work remains
subject to `MEMORY_SAFETY.md`; the original diagnostic replay remains 111/128.

Recorded [plan](plan.json), [all results](results.json), individual `case-*.json`
traces, and [source/resource validation](validation.json).
