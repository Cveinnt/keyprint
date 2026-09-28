# Natural source-grounded follow-up: frozen selection

The four long expansion stress tasks produced frequent factual failures even
without a watermark. Keep those failures; add a different task family to assess
whether the same problem persists when a model summarizes a substantial source
rather than expanding a short fact list. This does not redefine existing quality
acceptance, select successful ordinary responses or qualify detection.

The [Databricks Dolly dataset](https://huggingface.co/datasets/databricks/databricks-dolly-15k)
provides English instructions and context under CC-BY-SA-3.0. Snapshot
`bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a` is pinned with local file hashes.
Selection uses only category, source length and fixed source hashes. From 15,011
records, 377 meet the declared filters and represent 362 distinct source texts;
the first 16 salted source hashes are retained. Dataset answers are never used
to choose cases, prompt the model or establish ground truth.

## Frozen selection

[Manifest](selection.json) records all selected row IDs, instruction/source
hashes, length bounds, tie handling, fixed salt and selector source hash.
Changing reference answers or input file order does not change selected source
identities in the regression checks. Selection and all sixteen source-only
rubrics were frozen before generation. The [execution study](../source-grounded-run-2026-09-28/README.md)
now records the active 128-attempt run and its audit tooling.
Source text and derived prompts remain in private receipts, separate from the
MIT SDK. This public manifest does not relicense or republish dataset articles.

## Before inference

For each retained instruction, freeze the essential source facts needed to
answer it and any explicit output-format requirement. Do not remove an awkward
instruction or a potentially weak answer domain. Grade faithfulness to the
supplied historical passage, not current external-world knowledge. This public
dataset may have appeared in model training; call it fresh to this development
evaluation, never training-held-out.

Use the unchanged pinned Qwen3.5 experimental profile, all four existing keys,
ordinary/marked pairs, temperature 0.7, top-k 100 and a 768-token cap: 128 planned
attempts. Freeze the complete runner/source/identity plan before execution. Keep
every cap and failure; no retry, response replacement or post-generation repair.
Do not impose the previous artificial minimum length or three-paragraph format
when the original instruction asks for something different.

## Review and reporting

Freeze assistant ratings before revealing condition, key slot or raw counts.
Judge factual support of generated claims, preservation of essential task facts
and qualifiers, requested format and English separately. Accept harmless
paraphrase, not invented facts. Mark genuinely ambiguous wording before joining
metadata and report the sensitivity to accepting all such judgments. Human
ratings stay separate and blank until actually provided.

Report per-case and per-key paired outcomes, execution/cap failures and raw
matching/control diagnostics. There is no calibrated threshold for this new
profile. Do not report a missing verdict as a negative detection or borrow the
old tokenizer's threshold. Short correct answers can carry little signal.

This is an additional development study, not a powered noninferiority test,
universal semantic guarantee or release acceptance. The original multilingual
stress failures, historical registered harm bound and other compatibility,
reader-indistinguishability and serving gaps remain visible.
