# Native Qwen3.5 paired execution

All 32 scheduled ordinary/marked outputs completed through Keyprint's new
experimental Python path. All 10,239 sampled tokens reconcile with native byte
rendering, prompt IDs, runtime stopping policy and 32 distinct random-draw
transcripts. Zero execution errors. **Frozen assistant review still finds
factual and language failures; this is not quality, detector, serving or launch
acceptance.**

## Source-fact review

All 32 shuffled outputs were rated against each supplied fact, forbidden
additions, language and prose before revealing condition/key/count metadata.
Seven conservative flags were frozen with the original judgments. This is an
assistant development review, not independent human acceptance.

| Outcome | Ordinary / 16 | Marked / 16 |
| --- | ---: | ---: |
| Strict factual content | 2 | 0 |
| Requested language | 16 | 15 |
| Requested length and three-paragraph format | 9 | 9 |
| Strict full task | 2 | 0 |
| Content if all seven conservative flags pass | 6 | 3 |
| Full task if all seven conservative flags pass | 3 | 2 |

Clear failures include invented weekdays, tracking prerequisites and deployment
procedures. A marked French output says 20 GB is five times 5 GB; another gives
the price as above 18 rather than 18 EUR. One marked Japanese output inserts
the English word "itself". No output translates wholesale. Ordinary outputs
also invent facts, including a weekday and weather-related maintenance causes.

The comparison is descriptive, with four reused keys and four deliberately
demanding tasks. Low ordinary success creates a floor effect; these counts do
not establish a causal quality decline or a powered noninferiority result.
Passing every ordinary stress case is not a newly introduced launch criterion.
The adapter does not fix ordinary-model errors, nor excuse watermark-added harm.

[Frozen ratings](frozen-ratings.json), [pre-unblinding commitment](rating-commitment.json),
[complete results and sensitivity](fidelity-results.json), and
[every labeled output](SAMPLES.md) are retained. The human review page stays blank.
Raw matching-key counts aggregate to 75,049/149,190 ordinary and 79,232/148,620
marked; next-key controls are 74,659/149,190 and 74,742/148,620. These correlated
bit counts have no calibrated new-profile threshold or authorship verdict.

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
- [Full native replay](native-replay-results.json), under its
  [frozen replay plan](native-replay-plan.json): all 32 original attempts and all
  10,239 steps match bitwise model-head hashes, transformed-weight hashes,
  exact categorical draws and committed tokens. Direct native forwards,
  independent gap-first full-head filtering and scalar reference source policy
  bypass SDK generation, wider-head projection and sparse source execution.
  Native kernels, tokenizer binding and reference primitives are shared; this
  is not an independently implemented language model. No new random draws,
  replacement outputs or retries. Source-policy counters also match.
- [Review/replay checker tests](semantic-review-checks.json): 54 local checks
  pass, including scalar-versus-sparse source law, independent filter arithmetic,
  retained-draw replay, missing detection verdicts and deterministic source
  selection. No SDK or default behavior changed during this review/replay step.
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
to assess watermark-added harm. Matching/control counts were held back during
review and are now retained in the full results, with no borrowed thresholds or
detection verdicts.

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
PYTHONPATH=src python tools/replay_wide_mlx.py /private/new-wide-study \
  --model /path/to/pinned-snapshot --prior /private/prior-multikey-study
```

Use a new output directory. Future draws and outputs will differ; retained
receipts make each actual run inspectable. Do not relabel new output as the
recorded study or discard failed attempts.
