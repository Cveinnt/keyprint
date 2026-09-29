# Balanced multilingual study harness

The complete runner and analysis path are implemented. No model output has yet
been generated for the frozen 128-attempt evaluation bundle. This closes a test
infrastructure task, not a quality, detection or release requirement.

## Verified locally

81 tests pass across the balanced study, key-rank, frozen-input, inference and
session suites. They cover synthetic callback receipts, incomplete or altered
cohorts, preflight completion and cleanup, profile/protocol binding, fixed decoy
counts, owner/control score reconciliation, capped/failed denominator handling,
multilingual aggregation and rating commitments. An initial positive preflight
fixture failed because the study alternates arm order while the preflight does
not; the fixture order was corrected. No acceptance rule was weakened.

An actual CLI plan-only invocation validated the frozen bundle, pinned SDK and
runtime, and planned 128 attempts without importing MLX or loading a model.

- `tools/run_balanced_evaluation.py`: requires completed balanced preflight and
  both research memory guards. Runs every scheduled row once, retaining caps,
  errors, exact draw journals and returned text. No repair or translation.
- `tools/balanced_study.py`: reconciles all 128 retained outputs and blind review
  rows. Checks retained in-run sampler audits; does not independently replay
  native model heads or establish semantic quality.
- `tools/summarize_balanced_quality.py`: freezes all source-only assistant ratings
  and ambiguities before joining conditions. Reports strict and sensitivity
  results by language, component and paired task. Assistant review is not human
  approval or a powered noninferiority study.
- `tools/balanced_key_rank.py`: requires frozen ratings, uses all 199 decoys fixed
  before generation, reconciles owner/next-key scores, and reports both views at
  the already frozen conservative 0.01 rank threshold. Failed/capped responses
  remain unavailable, never clean negatives. No entropy or threshold search.

The source corpus, criteria, keys and schedule remain unchanged from
[the frozen-input receipt](../balanced-evaluation-inputs-2026-09-29/README.md).
Private keys and synthetic test artifacts are not included here.

## Resource status

Five system-memory-pressure readings over twelve seconds were normal. A
GPU/compression-inclusive process inventory nevertheless found the Android
emulator at approximately 8.82 GiB and a virtual machine at 4.35 GiB, with other
large processes still present. Free disk was approximately 6.72 GiB. These are
instantaneous measurements, not an allocation guarantee or leak diagnosis.
No unrelated process was terminated and no heavy job was restarted. The previous
balanced preflight's pressure stop, zero tokens and verified cleanup remain
retained. Another attempt requires headroom and fresh attempt directories.

## Release decision

Hold public launch, deployment and registry publication. The existing paced
candidate failed to establish useful detection or negligible quality impact;
new balanced math and passing harness tests do not replace actual results.
Next: establish headroom, complete bounded native preflight, execute the frozen
study, freeze all text ratings, then evaluate detection. Broader calibration,
serving and framework qualification remain separate requirements. The approved
public design and SDK defaults are unchanged.
