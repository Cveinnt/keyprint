# Paced candidate: fresh-key feasibility diagnostic

The fixed uniform 30-layer score clears a conservative random-key rank cutoff of 0.01 on **1/64 marked outputs** and **0/64 ordinary outputs**. Median ranks are 0.4075 marked and 0.5425 ordinary. All 128 outputs were available; none were excluded. This is not sufficient evidence of useful detection and does not qualify the candidate for promotion.

Before new control scoring, the plan fixed 199 fresh random decoy keys, the existing uniform score, conservative tie handling, a single cutoff, and every original output. Each rank is `(1 + number of decoy scores >= owner score) / 200`. Every path has the same event count under every key, so count ranking does not require a length-normalization assumption.

The optimized scorer reproduces all **256** saved owner/next-key event counts exactly, independently of its HMAC-template implementation. Fourteen tests compare compiled scores against original replay across repeated contexts, whitespace-equivalent classes, excluded tokens and split UTF-8 bytes; they cover conservative ties, exact cutoff boundaries, invalid counts and full-cohort retention. After completion, all 128 ranks were independently recomputed from the complete 199-by-128 decoy score matrix and all key/file hashes checked.

## Scope and decision

These are previously opened development outputs from sixteen reused English tasks and four reused generation keys. The same fresh decoy keys score multiple outputs. Ranks are conditional on recorded tokenizer paths, not arbitrary-text inspection. They are not production p-values, an independently estimated false-positive rate, independent Bernoulli trials, or a held-out power result. No threshold search, new generation, response replacement, semantic repair or quality rerating occurred.

A weak score could reflect poor extraction, poor weighting or insufficient information in the bounded distribution. This diagnostic does not distinguish those explanations. Next use the saved native base weights and exact integer marked distributions to measure conditional likelihood information and per-output concentration before allocating another model run. Such an oracle uses privileged probabilities and is not itself a shippable text detector. Keep its evidence separate from fresh, prospectively frozen detection/quality confirmation.

## Resource use

Tokenizer only; no model weights or inference loaded. All 199 decoys completed in 62.29 seconds under a 2 GiB external watchdog, with **546,997,232 bytes (0.51 GiB)** peak sampled physical footprint. All 224 pressure samples were normal; cleanup verified. No automatic restart. Frozen keys and detailed decoy score matrix remain in private receipts.

[Plan](plan.json), [score reconciliation](reconciliation.json), [results](results.json), [validation](validation.json). SDK defaults, public website, CI-off policy and launch hold remain unchanged.
