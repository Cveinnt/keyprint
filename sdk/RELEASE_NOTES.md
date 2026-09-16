# Keyprint SDK 0.0.4rc4 (unreleased)

- Add `keyprint demo`, `doctor`, `verify` and `generate`; human-readable output
  by default, full fixture reports with `--json`.
- Package the pinned local MLX example and an optional `mlx` dependency group.
- Add project URLs to distribution metadata and command-level regression tests.
- Preserve the frozen scientific bundle and legacy JSON entrypoints.
- Distribution/import renaming and SGLang/vLLM/hosted-provider integrations
  remain unfinished. This branch is not a new public compatibility claim.

# Keyprint SDK 0.0.4rc3

rc3 corrects packaging and public API documentation for the independent research
preview. It adds no efficacy, calibration, provider-attribution or authorship claim.

## Changes

- Require Python >=3.12, matching the unchanged `numpy==2.5.2` dependency. rc2
  incorrectly advertised Python 3.11. SciPy and tokenizers pins are unchanged.
- Export the existing `DurableJournal` as `keyprint_v3.DurableJournal`, making the
  journal required by `PublicCandidate.run_response` available through a supported
  import. The class identity and scientific caller are unchanged.
- Correct the outer third-party notice's distribution name, wheel-member paths,
  dependency pins and historical-inventory scope.
- Document supplied-logit/model usage, private journal handling, failure reports,
  unresolved trace hashes and CLI startup failures.

The frozen `_bundle`, `bundle-manifest.json` and historical `docs/` snapshots are
byte-identical to rc2. No core, tokenizer, support filter, sampler, model caller or
reporting facade was changed. The package wrapper version is `0.0.4rc3`; the
reporting facade deliberately continues to identify its unchanged rc2 code.
Previous releases and receipts remain preserved.

Reports retain the frozen facade's
`integration_status="public_python_rc2_facade_package_browser_acceptance_pending"`
and `all_reporting_surfaces_accepted=false`. Those fields describe that unchanged
facade's acceptance metadata, not current deployment or package/browser audit
status. This release does not convert them into a universal acceptance claim;
separately dated integration evidence must be assessed on its own terms.

- Core runtime: `0f78c82a544c496644d4e1c9b13809781a97baf4451b190fa8e2bcfac4f36477`
- Reporting facade: `7c1266a980dddaae26c91078b010c4aab6c24fca290dc96cbe23adb5b9bf4097`

## Evidence snapshots

The bundled architecture snapshot's **13/25** count is historical evidence at its
recorded date. The separate [current clue ledger](https://keyprint.vercel.app/clue-ledger.json)
reports **22/25** at rc3 preparation. Those later observations do not rewrite
the bundle, change original acceptance criteria or imply detector calibration.
Output-quality preservation (A02), blinded reader indistinguishability (A03) and
latency (A18) remain open. See [research evidence](https://keyprint.vercel.app/research/)
for current scope. A future ledger update must be read on its own terms.

The rc2 Unicode reporting correction is retained: completed decomposed-accent
text remains available as generated content even when literal scoring is
unavailable. It does not substitute a generation score, zero counts or a verdict.

## Migration and failure handling

Upgrade in Python 3.12 or newer and start a fresh process. Replace internal
`keyprint_candidate_v3_caller.DurableJournal` imports with
`from keyprint_v3 import DurableJournal`. Keep each attempt's journal in a
restricted directory and use a fresh path for subsequent attempts.

Public API operation failures return contextual reports or `PublicReportError`
where documented; constructor argument errors can raise ordinary exceptions.
CLI startup/import or argument errors produce stderr and a nonzero exit code,
possibly without JSON. A failed frozen-bundle hash check deliberately prevents
loading the reporting facade. This remains a fail-closed startup contract;
`verify` is a local bundle-integrity check, not a signature or authorship verdict.

## Validation

`tests/test_public_surface.py` exercises supported imports, journal/caller usage,
generated Unicode reporting, immutable core identities, CLI commands and the
nonzero/stderr corruption contract. Run it against an installed wheel from a
directory outside this source tree. Exact artifact hashes and execution results
are recorded separately in `audit/validation.json` after the build; that audit
file is not part of the frozen bundle.
