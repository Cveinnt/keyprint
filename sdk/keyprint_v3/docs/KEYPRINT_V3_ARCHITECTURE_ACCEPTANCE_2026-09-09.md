# V3 architecture acceptance: 13 of 25 scoped requirements

Independent review reproduced 13 accepted scoped requirements under one v3 runtime. This is architecture and integration evidence, not full public-clue conformance, human quality, calibrated detection or release acceptance. All 25 original criterion strings and hashes are unchanged.

Runtime: `0f78c82a544c496644d4e1c9b13809781a97baf4451b190fa8e2bcfac4f36477`.

## What changed

The complete float32 model head now passes through one bound, key-independent filter before either ordinary or marked sampling. The declared ordinary distribution uses mapped vocabulary, temperature 0.7, top-k 100 and a maximum scaled logit gap of 600. Marked sampling preserves every admitted alternative using the existing positive-integer categorical sampler. The same filter operates inside the public model-boundary step; the caller cannot submit a prefiltered float64 bypass.

A09 closes exact categorical reachability for the declared normalized law. A10 closes preservation of **post-filter** support. Raw tails outside the declared filter are intentionally excluded in both arms. This changes ordinary sampling and requires fresh empirical evaluation; it does not establish preservation of every finite raw model alternative. The caller casts model output to float32, so the boundary claim begins with that float32 head.

## Evidence

- 29 integration and caller tests passed independently.
- Fresh architecture audit reproduced all 13 acceptances with 40 bound source hashes verified.
- Six public supplied-model journals matched root execution byte-for-byte and passed hash-chain checks.
- Fourteen ordinary/marked support-policy cases covered boundary gaps, extreme float32 values and padded vocabulary.
- Ten additional invalid-input cases closed before random draws or commits.
- No private keys, model weights, detector fits or earlier empirical receipts were used for this audit.

| Clue | Requirement | V3 evidence |
| --- | --- | --- |
| A01 | Generation-time marking | Accepted, scoped |
| A02 | Output quality | Fresh empirical evidence required |
| A03 | Reader indistinguishability | Fresh empirical evidence required |
| A04 | No added characters | Accepted, scoped |
| A05 | No extra tokens | Accepted, scoped |
| A06 | No user identifiers | Accepted, scoped |
| A07 | Distributed choices | Accepted, scoped |
| A08 | Keyed local context | Accepted, scoped |
| A09 | Context-dependent randomness | Accepted, scoped |
| A10 | Post-filter model support | Accepted, scoped |
| A11 | SynthID family | Accepted, scoped |
| A12 | Matching-key detection | Fresh empirical evidence required |
| A13 | Attribution limits | Accepted, scoped |
| A14 | Length-dependent evidence | Fresh empirical evidence required |
| A15 | Factual sparsity | Fresh empirical evidence required |
| A16 | Proofreading fidelity | Component only |
| A17 | Exact code choices | Accepted, scoped |
| A18 | Serving overhead | Fresh empirical evidence required |
| A21 | Editing and rewriting | Fresh empirical evidence required |
| A22 | Involvement, not authorship | Component only |
| A23 | Translated output | Fresh empirical evidence required |
| A24 | Style detectors differ | Accepted, scoped |
| H02 | Model-level integration | Accepted, scoped |
| H05 | Copy/paste detection | Fresh empirical evidence required |
| H06 | Uncertain detector reports | Component only |

## Remaining work

Nine empirical requirements need this exact candidate: A02, A03, A12, A14, A15, A18, A21, A23 and H05. A16 has source-protection components but still needs end-to-end fidelity and independent semantic review. A22 and H06 need every public API, CLI, package, export and rendered state to preserve reporting context; a tested formatter alone is insufficient. Human ratings remain zero.

The six-response real-model smoke is independently reviewed and frozen, but **not activated**. Its saved preflight found 5.53 GiB free against the unchanged 8 GiB startup guard. It would test model integration only, not close quality or calibration requirements.

## Release boundary

The public browser and downloadable SDK remain v1 0.0.3rc1 with 11 scoped acceptances. No v2/v3 evidence has been relabeled as v1 evidence. V3 needs package integration, installed-package checks, same-profile empirical evaluation and final surface review before promotion. The restored frontend styling changes no scientific result.

## Exact receipts

- [Architecture audit](keyprint_v3_architecture_acceptance_2026-09-09.json), SHA-256 `42f8879b580a4d1d58b299463beb6168429680f3b0bb5416b639aa16964a9894`.
- [Independent integration review](keyprint_v3_integration_review_2026-09-09.json), SHA-256 `272197317d8b7c875a1936c3d5bda07fa5f427b6b57299fa707fae92f254a783`.
- [Independent scope report](keyprint_v3_integration_review_2026-09-09.md).
- [Frozen model-smoke plan](keyprint_v3_model_smoke_2026-09-09.plan.json).
- [Saved preflight](keyprint_v3_model_smoke_2026-09-09.preflight.json).
- [Reporting surface closure plan](KEYPRINT_V3_REPORTING_SURFACE_ACCEPTANCE_PLAN_2026-09-09.md).

## Follow-on reporting component

The standalone [v3 reporting contract](KEYPRINT_V3_REPORTING_CONTRACT_2026-09-09.md) now passes 19 tests independently. It covers four typed report kinds and preserves target/scorer identity, uncertainty and exact integer exports. It is not integrated into the frozen v3 runtime or public package, so A22/H06 remain component-only and acceptance remains 13/25. A later saved [preflight](keyprint_v3_model_smoke_2026-09-09.preflight-final.json) observed 4.30 GiB free, still below the 8 GiB startup guard; no model activation occurred.
