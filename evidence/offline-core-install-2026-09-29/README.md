# Core wheel: clean offline install verified

Built source commit `cefce38dc71ae5b1a885a42d34514d9f32a7a2d1` into
`keyprint-0.1.0a1-py3-none-any.whl`, then installed that wheel into a fresh virtual
environment outside the checkout. Package resolution used `uv --offline` and
the existing local cache, with one build/install worker. `PYTHONPATH` was removed
from child commands; the imported package path was verified inside the new
environment, not the source tree.

Verified on macOS arm64, Python 3.13.13:

- Wheel size: 2,373,918 bytes (2.26 MiB), excluding dependencies and model files.
- `uv pip check` passes; installed dependency versions are in `packages.json`.
- The `keyprint` console entry point, `--help`, `doctor --json`, `verify --json`
  and model-free `demo --json` all exit successfully.
- Doctor reports package imports and all 39 engine manifest files passing.
- All seven bundled HTML/CSS/JS/SVG assets match source bytes, including the
  progress-request cleanup fix.
- MIT license, Vincent Wu (Cveinnt) attribution and all four required license
  and notice files are present; their bytes match the checkout.
- No Torch, MLX, Transformers, FastAPI, OpenAI or Anthropic package was installed.
  No model was loaded. The demo is the declared A/B/C/D fixture, not prose.

The entire sequential build/install/check worker completed in 17.80 seconds.
Peak sampled process-group footprint was 315,442,424 bytes (300.83 MiB) under a
512 MiB budget. All 62 pressure samples were normal; cleanup was verified.
Docker and other applications were untouched.

This qualifies the current core wheel on this host. It does not qualify PyPI
installation, fresh network download speed, Python 3.12, other operating systems,
optional inference frameworks, detector accuracy, output quality or launch
readiness. Offline package resolution was enforced; general network activity
was not packet-captured or independently audited. No registry publication,
deployment, hosted CI or model request occurred in this validation pass.

Machine-readable evidence: `validation.json`, `resource.json`, `doctor.json` and
`packages.json`. The original wheel, command stdout/stderr and worker source are
retained in the private `low-memory-sdk-2026-09-29` receipt directory.
