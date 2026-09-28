# Source facts survive often, not reliably enough for a guarantee

This study uses the unchanged public SDK on twelve fresh synthetic tasks in
seven languages. All 96 generations finish at EOS, retain the requested language
and satisfy the requested length limit. **Factual preservation is a separate
question:** ordinary outputs pass 42/48 assistant content reviews; marked outputs
pass 36/48. These are descriptive results, not an estimate of causal watermark
harm or independent human acceptance.

| Measure | Ordinary | Marked |
| --- | ---: | ---: |
| All supplied facts preserved, no unsupported claims | 42/48 | 36/48 |
| Requested language retained | 48/48 | 48/48 |
| Requested length limit met | 48/48 | 48/48 |
| Completed without runtime errors | 48/48 | 48/48 |
| Matching-key diagnostic hits | 0/48 | 1/48 |
| Other-key diagnostic hits | 0/48 | 0/48 |

Paired content outcomes: 33 both pass, nine ordinary-only passes, three
marked-only passes and three both fail. There are nine exactly identical
ordinary/marked pairs, retained honestly. Distinct random draws do not guarantee
different wording on tightly specified tasks.

Nine conservative judgments concern ambiguous scope or omissions. If all nine
are treated as passes, the totals become **44/48 ordinary and 43/48 marked**.
The apparent gap is therefore sensitive to interpretation. Both primary ratings
and this complete sensitivity analysis remain available; no labels were changed
after conditions were revealed. See [sensitivity.json](sensitivity.json).

## Clear problems still present

- A marked Spanish answer changes planned Friday **dispatch** into planned
  Friday **delivery**, despite the source saying delivery is unconfirmed.
- A marked Japanese answer invents Friday for a date whose weekday/year was
  not supplied. One ordinary and one marked Japanese answer omit Tokyo time.
- German outputs in both conditions add unrequested customer-contact promises
  or omit explicit nonapproval status. Awkward refund/request grammar also needs
  human review; grammar alone was not automatically counted as factual failure
  where the surrounding nonapproval/no-payout clauses preserve the intended state.
- An ordinary French answer invents a false price-per-GB comparison. This is a
  base-generation failure too, not evidence that every bad sentence is caused
  by watermarking.

The largest conservative difference is Portuguese: “not dispatched sales” becomes
“not completed sales,” two potentially different business states. That judgment
is exposed for review, not disguised as an objective arithmetic failure.

| Task | Ordinary content passes | Marked content passes |
| --- | ---: | ---: |
| English incident status | 4/4 | 3/4 |
| French invoice | 4/4 | 4/4 |
| Spanish shipment | 4/4 | 3/4 |
| Chinese tentative meeting | 4/4 | 4/4 |
| German refund | 2/4 | 2/4 |
| Japanese maintenance | 3/4 | 2/4 |
| English release notes | 3/4 | 2/4 |
| Portuguese inventory | 3/4 | 0/4 |
| English validation boundaries | 4/4 | 4/4 |
| French plan comparison | 3/4 | 4/4 |
| Spanish publication consent | 4/4 | 4/4 |
| English ownership/blocked work | 4/4 | 4/4 |

Short source-grounded outputs are not a detector-power qualification. At the
unchanged thirty-layer weighted diagnostic rule (0.005 per key, two keys), only
one marked output hits. The nominal 1% family target remains uncalibrated for
deployment. Zero wrong-key/ordinary hits in this small set proves no error-rate
guarantee. Fidelity and detection must both be measured on longer tasks next.

## Inspect and review

Open [review.html](review.html) in a browser for all 96 outputs, supplied source
facts and prompts. It excludes sampler/key identities, scores and assistant
ratings. All human ratings start blank. Answers save locally under this dataset's
hash and export as JSON; uncertainty remains distinct from a pass. No external
service, model call, automatic submission or acceptance decision is involved.
The source file can also be served locally for consistent browser storage.

[All labeled samples](SAMPLES.md) reveal conditions. Use the review page first
if you want to form your own judgments without that mapping. Reader review here
concerns factual preservation; it is not the separate reader-indistinguishability
experiment or evidence that the historical A03 requirement is closed.

## Method and verification

- Twelve new task definitions in `tools/fact_fidelity_cases.json`, frozen before
  inference. Every source fact and forbidden addition is listed. English, French,
  Spanish, Chinese, German, Japanese and Portuguese; one pair per task/key.
- All four existing September 24 keys, including the known failing key, retained.
  Condition order alternates by key and task. No retries, output replacement,
  key selection, translation, rewriting or hidden prompt repair.
- Pinned Qwen3-8B MLX revision `545dc4251c05440727734bcd94334791f6ab0192`, temperature
  0.7, top-k 100, 256-token cap. The public `Keyprint.generate` API is used.
- Fact-level, unsupported-claim and language judgments were frozen before opening
  condition/key mapping or scores. This is assistant review with known task
  design, not independent human review. Format checks are mechanical and separate.
- All **6,388 committed tokens**, rendered bytes, journal chains, assigned
  conditions and model-call records reconcile. All 96 draw transcripts differ.
  Hash chains prove consistency, not independent authenticity.
- Current source and installed package match all 90 files in the prior wheel.
  All 39 frozen engine files pass. 29 local evidence/review tests pass.
- Browser QA uses isolated fixtures for ratings and the real page for blank-state,
  selection and layout checks. Desktop 1440×1000 and mobile 390×844 pass; export,
  uncertainty handling, reload persistence, keyboard controls and inert generated
  markup are exercised. QA fixture labels never become human study evidence.

Reproduce with the pinned model and original private keys:

```bash
PYTHONPATH=src python tools/validate_fact_fidelity.py \
  --model /path/to/545dc4251c05440727734bcd94334791f6ab0192 \
  --prior /path/to/multikey-prose-2026-09-24 \
  --output /new/private/fact-study
python tools/build_fact_review.py \
  /new/private/fact-study/public/blind-review.json /path/to/review.html
```

Freeze fact-level ratings and their hashes before running
`tools/summarize_fact_fidelity.py`. [Plan](plan.json), [rating commitment](rating-commitment.json),
[all judgments](frozen-ratings.json), [results/audit](results.json) and
[SDK integrity](sdk-integrity.json) retain the scope and evidence.

No SDK source, sampler, core dependency, production page, hosted CI or release
status changes. Do not pool this new task set with earlier studies as though
their different rubrics and reused keys formed one independent benchmark.
The historical quality criterion and 25-clue ledger remain open where previously
open. Clear failures require investigation; borderline wording requires honest
review, not silent promotion or a promise of perfect semantic preservation.
