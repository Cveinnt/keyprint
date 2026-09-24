# Installed wording comparison, September 24

The shared local/exported viewer now places **Highlight differences** above the
response panels. It highlights exact wording differences between independent
samples, not watermarked words. Identical outputs receive an explicit message
and no highlights. Disabling it restores the selected reading mode.

The renderer preserves every source character, keeps generated HTML inert, and
uses a bounded alignment table. Long dissimilar passages fall back to a disclosed
block comparison without truncation. Existing cream/rust styling is retained.
No runtime dependencies, generation calls or sampler changes are added.

Validation: 65 affected Python tests and all 28 JavaScript tests pass locally.
One existing dependency deprecation warning appears in the Python TestClient.
The first JavaScript run exposed a missing DOM method in the test double; its
contract was corrected and both recorded/live paths then passed.

A fresh Python 3.13 environment installed the built wheel and passed package
diagnostics. All 90 package files match source, wheel and installation. Exporting
the existing gallery through the installed public API preserves its data exactly.
See [install receipt](installed-audit.json) and [diagnostics](doctor.json).

Browser checks covered desktop and 390px mobile, keyboard toggling, failure-note
retention across gallery switching, exact-text preservation and custom-prompt
entry. The installed MLX playground generated its prefilled pair and a custom
approval-note pair. The latter was identical across conditions; the control
correctly showed zero highlights. Refresh restored the pair without creating
another request. No browser warnings or errors were observed.

[Both live pairs](live-comparisons.json) retain exact public traces and outputs;
all four token-ID sequences and reconstructed texts match private reports.
Keys, session URLs and raw journals are excluded. [QA receipt](browser-qa.json)
records the checked interactions; `sha256.json` commits the JSON files.

This qualifies the changed viewer on this Mac with cached Qwen/MLX. It does not
qualify fresh model download, other operating systems, general output quality,
or other serving runtimes. Production remains unchanged.

## Landing first-load improvement

The separate website's 144-pair data set previously shipped inside initial
JavaScript. It now loads when the reader approaches the comparison section.
Initial JS fell from 1,004.21 kB to 313.95 kB, or 237.57 kB to 86.70 kB gzipped.
All 144 records are preserved in a separate 689.83 kB asset. This is a byte-size
improvement, not a measured page-speed or LCP result; images/fonts are unchanged.

The production build and four site checks pass. Browser/server observations
confirm the pair asset is absent from initial requests and loads through the
Compare link. Task filtering retains 48 reading examples. A fresh `#outputs`
navigation reaches the section. A simulated missing-asset response exposes Retry;
restoring the asset and retrying recovers all 144 options without losing the
custom draft above. That intentional 404 is excluded from normal console checks.

The website lives outside this SDK repository. The local source was updated;
[retained patch](landing-deferred-comparisons.patch), source hashes and
[load receipt](landing-load.json) record that change. The local site's recorded
viewer assets were also refreshed from the installed SDK export. The selected
full-page design and recorded outputs remain intact. Nothing was deployed.
