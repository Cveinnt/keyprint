# Recorded gallery and installed-package checks

Three examples use exact SDK data in the same viewer: an existing explanation
pair and two newly generated Qwen/MLX pairs (email and French). All six token
traces reconstruct the outputs exactly. All generated samples were retained.

The marked email invents “the park around noon” despite the prompt forbidding a
time or location. The gallery shows that constraint failure beside the sample.
French outputs remain French. These observations do not establish a quality
rate, causal watermark harm, reader indistinguishability or acceptance of A02.

Fresh Python 3.13 environment installed the built wheel, passed `keyprint doctor`,
and exported the full gallery using the installed public API. All 90 package
files match both source and installed files. This is a local wheel installation,
not a public registry or fresh-machine inference test. No new model downloaded.

59 affected Python checks, 23 JavaScript checks and four site checks passed.
Browser QA verified example switching, flagged failure, custom-prompt handoff,
backend-specific recipes and 390px mobile layout. Screenshots remain in local
receipts. A gallery request fetches only local static data, never a model API.
No new dependencies, public deployment or social dispatch. CI stays disabled.
