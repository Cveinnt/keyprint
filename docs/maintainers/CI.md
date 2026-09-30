# Repository checks and inference qualification

## Small, model-free checks

`community-checks.yml` is manual-only pending the owner's decision about
automatic CI. It uses one standard Ubuntu 24.04 runner with a five-minute job
limit, read-only repository permissions and pinned checkout/setup actions.
There are no model downloads, hosted inference calls, package installations,
publication steps or self-hosted runners. Node's heap is capped at 128 MiB;
the UI suites run sequentially.

Run the same checks locally from the repository root:

```sh
python tools/check_community.py
node --max-old-space-size=128 --test --test-concurrency=1 tests/test_gallery_ui.cjs tests/test_playground_ui.cjs tests/test_reader.cjs tests/test_token_explorer.cjs tests/test_wording_diff.cjs tests/test_share_ui.cjs
```

These check local documentation targets, Python syntax without importing the SDK,
recording hashes and exact trace/text reconstruction, the downloadable gallery,
and viewer behavior. They do not establish wheel installation, native integration,
real inference, detector accuracy, semantic fidelity or production readiness.

## Heavy workflows stay separate

- `test.yml`: full SDK/reference wheel matrices, client tests and real CPU inference.
  Manually disabled. Its title explicitly identifies full qualification.
- `sglang-inference.yml`: actual native SGLang CPU comparisons. Manually disabled.
- `publish-pypi.yml`: legacy publisher for `keyprint-research-v3==0.0.4rc3`.
  Disabled; it must not be used to publish the new `keyprint` distribution.

The old failed jobs retained in Actions did not execute their test steps. GitHub's
annotation reported an account-payment or spending-limit problem. Preserve those
records; deleting failures is not a fix or evidence of passing checks.

## Branch and ownership conventions

The reviewed community release and its contributors remain separate from the
unvalidated work on `launch-readiness-sdk`. Never merge that branch just to tidy
the branch list. Published tags are immutable; root `sdk/` preserves the original
reference and notices. `CODEOWNERS` routes review to Cveinnt without changing
historical authorship. Merged feature branches may be cleaned up automatically;
unmerged work stays available.
