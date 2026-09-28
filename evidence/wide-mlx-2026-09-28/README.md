# Native Qwen3.5 paired execution

All 32 scheduled ordinary/marked outputs completed through Keyprint's new
experimental Python path. All 10,239 sampled tokens reconcile with native byte
rendering, prompt IDs, runtime stopping policy and 32 distinct random-draw
transcripts. Zero execution errors. **Semantic review is pending; this is not
quality, detector, serving or launch acceptance.**

## What changed

The wider-head sampler selects the highest-ranked finite logits, restores
ascending original token order, delegates bounded arithmetic to the unchanged
reference implementation, then restores full-head IDs. Tests compare bitwise
probabilities and integer draw transcripts, including original IDs above the
previous vocabulary limit. Default vocabulary guards remain unchanged.

The explicit NFC binding retains the original serialized tokenizer and all
special-token exclusions. It does not normalize generated output. Literal
inspection rejects text changed by tokenizer normalization, including decomposed
accent examples. Model input uses its declared NFC tokenizer. The runtime EOS is
248046, explicitly checked against the installed loader; nested model metadata
instead names 248044. Every request owns and releases its native cache.

## Evidence and scope

- [Plan](plan.json): unchanged four longer factual tasks, all four existing keys,
  two conditions, 1,024-token cap, temperature 0.7, top-k 100. No retry, replacement,
  output repair or key selection. SDK and script hashes frozen before generation.
- [Identity](identity.json): model assets, runtime versions, profile, normalizer,
  stopping policy and source hashes. No previous empirical acceptance transfers.
- [Receipt audit](receipt-audit.json): all attempts, exact bytes, tokenizer decode,
  prompt, token count, EOS, RNG rejection transcript and commit order. Hashes and
  journals establish recorded execution consistency, not independent re-execution
  of model heads or proof of model factual correctness.
- [All shuffled samples](blind-review.json) and [offline review](review.html):
  every output retained, conditions/keys/counts hidden; blank human ratings.
- [Local checks](local-checks.json): 291 focused tests pass across two existing
  environments, three optional checks skipped. Initial fixture-type and build
  frontend errors retained and explained; no inference was retried.
- [Wheel check](wheel-check.json): isolated no-dependency installation; all 78
  recorded Python sources match and 39 frozen engine files pass integrity.
  Uses existing pinned dependencies, not a clean-environment install qualification.

This is the content-pinned `mlx-community/Qwen3.5-9B-4bit` snapshot
`8b2b98c00a6b4d291155e4890773ca8f769aee53` on local Apple Silicon. MLX 0.32.2,
MLX-LM 0.31.2 and Transformers 5.16.1 are required. Text-only Python API pilot:
no inherited provider-client, JSON grammar, tools, reasoning, streaming, batching,
concurrency, other-model or framework qualification. Production site and CI were
not changed. SDK publication and publicity remain held.

The separate ordinary baseline already contains factual errors. Successful
execution does not imply language/meaning preservation or remove the obligation
to assess watermark-added harm. Raw matching/control counts are retained
privately until review, with no borrowed thresholds or detection verdicts.

## Reproduce locally

From this source checkout with the pinned MLX dependencies and model already
available, provide the prior development key set locally. Do not publish keys or
raw draw journals.

```sh
PYTHONPATH=src python tools/validate_wide_mlx.py \
  --model /path/to/pinned-snapshot \
  --prior /private/prior-multikey-study \
  --output /private/new-wide-study
PYTHONPATH=src python tools/audit_wide_mlx.py /private/new-wide-study \
  --model /path/to/pinned-snapshot
```

Use a new output directory. Future draws and outputs will differ; retained
receipts make each actual run inspectable. Do not relabel new output as the
recorded study or discard failed attempts.
