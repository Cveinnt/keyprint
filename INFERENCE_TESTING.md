# Actual inference compatibility checks

Run the same six prompts through ordinary and marked sampling, retaining both
texts, completion reasons, timings and matching/other-key diagnostics. Cases
cover science, a constrained email, negation, Spanish, French and strict JSON.
Independent samples are compared descriptively; differences are not a causal
estimate of watermark quality loss.

## Reproduce locally

Install the wheel with the backend, server and clients extras, and download the
pinned model listed in README.md. Then run:

```sh
python tools/validate_compatibility.py --backend transformers \
  --model models/smollm2 --output private-inference-run --http-client
```

For the pinned Qwen MLX backend, use `--backend mlx --model models/qwen3-8b-4bit`.
The run directory must not exist. It contains private keys and journals; do not
commit or upload it. Only `public/` contains exportable synthetic-case texts and
reports. Open `public/comparison.html` to read ordinary and marked text side by
side. Every attempt is retained. A failed generation fails the engineering run;
truncation, missing literals and incorrect JSON are reported separately.

The HTTP check runs an actual OpenAI SDK request through a TCP socket to a local
model. It verifies generated text against the private report, exact idempotency
replay, a single model attempt and rejection of unsupported options. An actual
Anthropic SDK request confirms that the Messages endpoint is unsupported (404).
Provider-object rewrite checks construct real SDK response objects from a fixed
synthetic document and run the local model. They are not hosted GPT/Claude calls.

To exercise client timeout recovery separately:

```sh
python tools/validate_server_lifecycle.py --model models/smollm2 \
  --output private-lifecycle-run
```

This check deliberately pauses the first accepted request before inference so a
real OpenAI SDK client times out. It verifies busy/conflict rejection, releases
the worker to generate real text, and recovers the exact result with the same
request ID without a second generation. A fresh request then checks worker reuse.
The test also checks graceful shutdown. This validates process-lifetime recovery,
not latency, crash recovery or model preemption. Only `public/` is exportable.

## CI

The Transformers job builds and installs the wheel, downloads
`HuggingFaceTB/SmolLM2-135M-Instruct` at revision
`12fd25f77366fa6b3b4b768ec3050bf629380bac`, then runs inference with Hub access
disabled. CPU PyTorch avoids installing an unused CUDA stack. A pinned upload
artifact action retains only `public/` for 14 days, including failed-run reports.
The job summary separates engineering failures from text-screening flags.
The same installed-wheel job runs the timeout-recovery check and retains its
public lifecycle report alongside the paired-text comparison.

A green job means the bounded runtime and protocol contracts passed. It does
not mean that generated prose is correct or safe to publish. This small model
is an integration fixture, not a quality benchmark or recommended rewrite model.
There are no hosted-provider credentials or API calls in this job.

Rewrite screening checks exact numbers, URLs, email addresses and full weekday
names in English, French and Spanish. This catches observed Friday-to-Monday
drift even when the numeric time stays intact. It does not parse dates, cover
abbreviations/relative dates, or verify which event a weekday belongs to.
Passing these checks still yields `needs_review`, never automatic acceptance.

## Longer-text detection feasibility

```sh
python tools/validate_detection_screen.py --backend mlx \
  --model models/qwen3-8b-4bit --output private-detection-screen
```

The script writes its complete plan before loading the model. Twelve ordinary
calibration prompts determine the maximum observed length-normalized bit excess,
`(2 * ones - trials) / sqrt(trials)`. It freezes that threshold before generating
ordinary/marked pairs for twelve different held-out prompts. Each response is
also inspected under an independent control key. A result exceeds the threshold
only with a strict greater-than comparison; ties do not count.

English, French and Spanish prompts request about 220 words, with a 512-token
cap. Truncated outputs and failures remain in the report. Calibration failures
stop the experiment; held-out failures remain in the attempted denominator.
The fresh output directory contains private keys and journals. Only `public/`
is exportable, including the plan, frozen threshold, every generated text and
summary counts. Do not choose a new threshold after reading held-out results.

This small synthetic screen asks whether the current statistic shows useful
separation at this length. It cannot qualify rare false positives, short inputs,
edits, unrelated domains, different keys/models or production detection power.
The statistic has no normal-distribution or authorship-probability interpretation.
It is not a quality experiment; prose still requires review.

## Diagnose saved MLX signal capacity

```sh
python tools/analyze_mlx_capacity.py --study private-detection-screen \
  --model models/qwen3-8b-4bit --output private-capacity-replay \
  --cases heldout-00-ordinary heldout-00-marked heldout-01-marked
```

This diagnostic replays existing generations using their original prompts and
saved random-bit transcripts. It requires matching sampler identity, exact raw
logit hashes, base/prepared probability hashes, selected tokens and random draws.
It fails on disagreement; it never relaxes a mismatch or draws fresh randomness.
The original generation files stay unchanged. Per-step diagnostics remain private;
only aggregate `public/` reports may be exported.

Entropy, expected bit lift, distribution distance and per-token log ratios help
distinguish limited token choice from an inefficient text statistic. A token can
have many one bits while providing no watermark information if its probability
is unchanged. The probability ratios require model weights, original prompts and
private execution state: they are an oracle diagnostic, not a text-only detector.
This analysis uses already-opened data. It cannot validate a new detector or
replace fresh confirmation after a proposed change is frozen.

## Compare and confirm a text-only score change

```sh
python tools/compare_score_baselines.py --study private-detection-screen \
  --output private-score-development
python tools/validate_score_confirmation.py \
  --comparison private-score-development/comparison.json \
  --key private-detection-screen/owner.key --model models/qwen3-8b-4bit \
  --output private-score-confirmation
```

The first command compares uniform weights with the linear 10-to-1 layer weights
documented in [DeepMind's SynthID Text detector](https://github.com/google-deepmind/synthid-text/blob/main/src/synthid_text/detector_mean.py).
It independently replays literal event bits and requires the original uniform
counts/statistics to match. Scores use length normalization, without a p-value
or confidence interpretation. These already-opened texts are development data;
an improvement here is not confirmation.

The second command freezes both weights and their previously selected thresholds
before generating 12 ordinary/marked pairs from different prompts. It keeps the
original generation key and model, introduces a fresh other-key control, retains
every attempt and applies strict greater-than decisions without tuning. This
tests a proposed scoring change on fresh text; its small, same-key/model scope
still cannot certify deployment error rates or generalize to new domains.

## Public human-text false-positive screen

```sh
python tools/validate_null_corpus.py \
  --source databricks-dolly-15k.jsonl \
  --baseline private-score-development/comparison.json \
  --output private-null-corpus
```

Download `databricks-dolly-15k.jsonl` from
[Databricks Dolly](https://huggingface.co/datasets/databricks/databricks-dolly-15k/tree/bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a).
The runner requires that revision's exact file checksum. The dataset is copyright
2023 Databricks, Inc., with some Wikipedia contributions, under CC BY-SA 3.0.
Contributors were instructed not to use generative AI; this is source provenance,
not independent authorship verification. Public reports retain indices/hashes and
scores, not copies of the responses. Keep attribution with exported reports.

Before scoring, the runner groups transitive exact normalized duplicates across
instructions, contexts and responses, then deterministically selects 1,000
responses of 100–400 whitespace-separated words. It uses 500 for calibration and
500 for validation, with two fresh independent keys per split. Each observation
is one document's maximum across its two keys. The 2,000 individual key scores
are not 2,000 independent documents. It freezes both 1% empirical thresholds
before held-out scoring. Missing scores remain unavailable, not negative results.

The reported 97.5% one-sided exact binomial upper bounds assume IID documents;
related subjects/authors, near duplicates and changed keys limit that assumption.
Neither these bounds nor the calibration rank confer a deployment guarantee.
Results cover this English corpus and length range only. Detection power must
be tested separately on fresh marked outputs under the frozen thresholds.

```sh
python tools/validate_corpus_power.py --source databricks-dolly-15k.jsonl \
  --null-study private-null-corpus --model models/qwen3-8b-4bit \
  --output private-corpus-power
```

The follow-up chooses the first two eligible source tasks in each of six declared
categories, using only source metadata and the frozen split, never null scores.
It generates ordinary/marked pairs with the two held-out keys, alternating order
and key by task. Source instructions and reference contexts remain unchanged;
there is no forced response length. Outputs outside the null corpus's word range,
errors and truncations remain visible. Matching-key sensitivity, wrong-key hits
and ordinary two-key exceedances are separate counts. No thresholds are retuned
and no semantic-quality acceptance is inferred from detection results.

## Completion-budget diagnosis

```sh
python tools/validate_completion_extension.py --study private-corpus-power \
  --model models/qwen3-8b-4bit --output private-completion-extension
```

This experiment selects every original 512-token response ending at the limit.
It replays each saved prefix, requiring exact raw logits, sampled tokens, complete
probability records and random-bit requests/values to match before permitting
fresh randomness beyond token 512. The sole generation change is a 1,024-token
cap. Original outputs remain unchanged, and each extended output retains its
prefix audit, completion, usage and errors. Already-complete source outputs are
not regenerated. Selection is based only on completion.

This is controlled replay of already-observed paths, not fresh independent
quality evidence or production resume/crash recovery. Timings include replay and
cannot be reported as continuation latency. The observer is instance-local and
returns the original sampler results; frozen engine source is unchanged.

The playground now offers caps up to 1,024 tokens without changing its short
prefilled example or automatically retrying truncated answers. Its status tells
users when an answer reached the limit and how to request a new pair. Inspection
accepts up to 16,000 characters, matching the SDK's character bound; generation
prompts remain capped at 6,000. Token/profile bounds can still make a pasted
text's diagnostic unavailable, which must not be shown as zero signal.

## Conditional-layer detector development

`tools/develop_layer_detector.py` fits a research-only conditional layer density
from 24 previously opened marked responses, then evaluates previously opened
corpus controls and generated texts under different keys. It draws on
[SynthID Text's autoregressive layer likelihood](https://github.com/google-deepmind/synthid-text/blob/addb4a158143c7c6851a1308f78b89fceed59683/src/synthid_text/detector_bayesian.py),
with an independent NumPy/SciPy implementation. Coefficients, ridge penalty,
mixture fractions, source hashes and a fixed cutoff are retained. No prompt,
model probabilities or private journal is needed for text scoring.

Each conditional binary density is normalized. Under independent fair null
bits, its likelihood ratio has mean one. The implementation averages whole-text
evidence across five fixed contamination fractions; it does not maximize across
them. A two-key Markov/union cutoff is conservative under this idealized null.
It is not a posterior or fixed-key deployment guarantee. Exhaustive small-space
tests check normalization and tail bounds. The September 19 development run
produced 0/500 null hits but only 5/12 marked detections. Keep that failed power
result; do not promote the trained scorer into the SDK.

## Fresh weighted-reference validation

The original linear 10-to-1 score can also be evaluated against its numerical
weighted-Bernoulli null, instead of fitting a rare-tail cutoff to 500 documents.
Integer weights `290 - 9*i`, for layers `i=0..29`, preserve the original score.
For `n` eligible events, the centered sum's characteristic function is
`product(cos(weight*t))**n`. The method follows the general
[DFT characteristic-function approach](https://arxiv.org/abs/1702.01326).

`tools/weighted_null.py` inverts that function on a finite lattice. It adds a
Hoeffding alias bound and a `1e-9` numerical margin, refuses failed probability
mass checks, and returns one for nonpositive statistics. The margin is not a
formal proof of floating-point accuracy. Tests compare exhaustive enumeration,
an independent binomial calculation and a doubled grid. Validation doubles the
grid for every positive observed score and refuses differences above `1e-10`.

```sh
python tools/validate_weighted_null.py --source databricks-dolly-15k.jsonl \
  --prior-study private-null-corpus --output private-weighted-null
python tools/validate_weighted_power.py --source databricks-dolly-15k.jsonl \
  --null-study private-weighted-null --model models/qwen3-8b-4bit \
  --output private-weighted-power
```

The null runner freezes 500 previously unused exact-deduplicated source groups
and two new keys before scoring. Its sole primary decision is
`2 * min(two key reference tails) <= .01`. Uniform binomial results are a
descriptive secondary comparison, not a second opportunity to declare a hit.
The first fresh run produced 5/500 weighted false hits, with an IID-only 97.5%
upper bound of 2.32%. This does not establish a deployment bound below 1%.

The power runner freezes twelve new source tasks across six categories, uses
unchanged prompts, alternates ordinary/marked order and key assignment, and
allows 1,024 tokens. All outputs, truncations and answers outside the null
corpus's 100–400-word range remain in the result. The SDK's literal counts must
agree with independent bit extraction. Neither runner changes SDK detection
verdicts, establishes semantic quality, or transfers to arbitrary models/keys.

The completed fresh run retained all 24 outputs with zero runtime errors:
8/12 marked texts detected, 0/12 ordinary texts flagged and 0/12 marked texts
flagged under the wrong key. Four outputs reached the 1,024-token cap.
Descriptively, detection was 1/5 for marked answers within the original
100–400-word control range, versus 7/7 above that range. These tiny subgroups
are not length-specific power guarantees; do not discard the four misses.
Source-based review also found a changed league timeline and an unsupported
renaming claim in the marked Walton summary. In the Edinburgh pair, the marked
answer transferred the city's nickname to the university, while the ordinary
answer added an incorrect English regnal number. These are retained quality
failures, not a causal estimate of watermark-induced degradation.

## Article-length null controls

`tools/prepare_wikitext_controls.py` prepares a second source without inspecting
scores. Both training shards of `Salesforce/wikitext`, revision
`b08601e04326c79dfdd32d625aee71d232d685c3`, configuration
`wikitext-103-raw-v1`, are SHA-256 pinned. Top-level articles span shard
boundaries; section headings remain in their source text. Exact normalized
titles and 300-word openings are grouped transitively. One representative per
group is hash-ranked, with alternating 300/600-word prefixes from the same pool
of articles at least 600 words long. The default selects 5,000 per length.
Preparation found 29,444 articles, 28,362 long enough, and 28,088 exact groups.

```sh
# Use a separate data environment; pyarrow is not an SDK dependency.
uv run --with 'pyarrow==25.0.1' python tools/prepare_wikitext_controls.py \
  --source-dir pinned-wikitext-shards --output private-wikitext-controls
python tools/validate_wikitext_null.py --controls private-wikitext-controls \
  --prior-study private-weighted-null --output private-wikitext-null
```

The validator freezes two fresh keys and keeps the same sole primary rule:
`2 * min(two key reference tails) <= .01`. It checks the doubled FFT grid on
every text/key replay and retains all failures. Each article is one observation
across both keys and belongs to only one length stratum. Per-length and pooled
97.5% one-sided binomial upper bounds are descriptive under IID assumptions,
not simultaneous confidence bounds or fixed-key deployment guarantees. Shared
authors, topics and near duplicates may remain dependent. Do not combine this
new-key null screen with earlier power counts into an accepted deployment rate.

Source spacing, punctuation and `@-@` artifacts are preserved; no detokenization
or score-based cleanup is performed. This is an English encyclopedia corpus,
not natural chat, multilingual coverage or independent authorship verification.
The [pinned source card](https://huggingface.co/datasets/Salesforce/wikitext/blob/b08601e04326c79dfdd32d625aee71d232d685c3/README.md)
has conflicting license labels: CC BY-SA 3.0/GFDL metadata and CC BY-SA 4.0 prose.
The manifest retains this discrepancy and attribution to Wikipedia contributors
and WikiText's authors. Source passages stay outside `public/`; derived scores,
article metadata and provenance are exportable research artifacts.

The completed September 19 screen retained all 10,000 articles and 20,000
text/key replays, with zero errors. It produced 115 false hits (1.15%; IID-only
97.5% upper bound 1.38%). At 300 words: 70/5,000 hits (upper 1.77%). At 600 words:
45/5,000 (upper 1.20%). Maximum doubled-grid difference was `1.34e-12`. These
results do not establish a deployment bound below 1%. Collecting more samples
at a nominal 1% rule is not a sound strategy for forcing its confidence upper
bound below 1%; a stricter operating rule would need fresh confirmation.

## Learned marginal weights: negative development result

`tools/develop_layer_weights.py` tests one fixed recipe for using layer signal
more efficiently. It fits Jeffreys-smoothed marginal probabilities on the same
24 previously opened marked training responses (7,206 events), clips to
`[.5,.75]`, converts to positive log odds, then rounds to integer weights with
maximum 256 and minimum 1. The ceiling is a regularizer, not an empirical claim.
Evaluation keys differ from training keys. Training events may overlap across
documents, so the fit does not imply independent-trial parameter confidence.

`tools/learned_layer_weights.py` evaluates the fixed fitted weights against the
same ideal fair-bit null. It removes the common integer factor before FFT
inversion; this avoids unnecessary lattice size and roundoff without changing
the score ordering or null tail. Tests compare complete small null spaces and
an independent equal-weight binomial calculation. No SDK code changes.

```sh
python tools/develop_layer_weights.py \
  --training private-initial-screen private-score-confirmation \
  --power private-corpus-power private-weighted-power \
  --null-old private-null-corpus --null-fresh private-weighted-null \
  --source databricks-dolly-15k.jsonl --output private-learned-weights
```

The primary two-key family threshold is `.005`; `.01` is descriptive only, not
another chance to declare a hit. At `.005`, learned and original linear weights
both detected 10/12 marked texts in the first opened power set and 8/12 in the
second. Neither flagged an ordinary output. Both flagged 5/1,000 opened null
controls. At `.01`, learned weights had 11/1,000 null hits versus the original
14/1,000, but no improvement in marked detections. All 48 generated outputs and
1,000 ordinary controls were retained with zero scoring errors. The learned
weights did not recover the short-answer misses and are not promoted into the
SDK. This is development on opened data, not fresh validation or a new acceptance.

## Framework pilots

### Cooperative cancellation

```sh
python tools/validate_cancellation.py --backend transformers \
  --model models/smollm2 --output private-cancellation-transformers
python tools/validate_cancellation.py --backend mlx \
  --model models/qwen3-8b-4bit --output private-cancellation-mlx
```

This check uses a real OpenAI client and local HTTP server. It holds a deliberate
barrier after a real model call and at least one committed token, requests
cancellation twice, confirms the worker stays busy, then releases the barrier.
No model output or sampling draw is substituted. The check requires no further
model calls or token commits, correct partial usage, replayable terminal HTTP
410 without another attempt, successful fresh inference on the same worker,
late-cancellation preservation of success, and graceful shutdown. Journals and
keys remain private; `public/cancellation.json` contains the safe outcome and
next generated text. This establishes lifecycle behavior, not kernel preemption,
latency, quality or production framework cancellation. CPU CI runs the same
check from the installed wheel.

The September 19 fresh-wheel runs passed for Transformers/SmolLM2 and MLX/Qwen.
Each cancelled attempt retained one committed token. The following 32-token
generation succeeded on the same worker, ended at its length cap, and replayed
without another attempt. Both reports retain that truncated next text. The
installed-wheel suite passed 208 tests, including cancellation before work,
between calls, during a random draw, terminal races and unrelated model errors.

The local playground also supports authenticated `/api/cancel` with the active
experiment's `Idempotency-Key`. Stop remains pending until generation or the
current prefix inspection returns. Terminal HTTP 410 is replayable; it does not
replace the last completed pair or edited-text measurement. Browser checks on
September 19 exercised actual Qwen generation, keyboard Stop, refresh during
work, and custom text. A long inspection finished before Stop became available;
that timing-sensitive test was retained as failed. A separate controlled run
paused after an actual SDK inspection to verify pending-stop refresh, unchanged
prior measurements, no subsequent prefix work, and successful fresh inspection.
Desktop (1280×1000) and mobile (390×844) passed that controlled flow. This pause
tests lifecycle behavior, not cancellation latency or throughput. The browser
checks used isolated headless Chrome through Playwright; no provider keys or
personal browser profile were used.

### Native callback comparisons

`tools/validate_native_cases.py` runs the same cases as ordinary/marked pairs in
pinned CPU vLLM or SGLang environments. It compares actual final token IDs with
hash-chained selection journals; text retokenization is not substituted for host
commit verification. Each runtime is separate from the installed-wheel test,
with different dependency versions and explicitly experimental support.

The SGLang Dockerfile exposes a reusable `runtime` target before its historical
pilot and receipt-export stages. It retains the documented ARM NUMA workaround.
Neither framework pilot qualifies GPU serving, streaming, cancellation,
speculation, cache reuse or arbitrary model/tokenizer families.

The `comparison-receipts` target runs the full paired native-case script and
exports its receipts directly. The revised build fetches Git blobs on demand,
does not retain uv caches in image layers, and builds the patched CPU kernel
once with `--no-build-isolation` against upstream's installed PyTorch dependency.
This addresses the earlier second-PyTorch-environment disk failure. It does not
change the NUMA workaround or establish inference success: check both exported
exit-code files, comparison outputs and token journals. Resolved dependency
versions remain recorded per build and are separate from distribution pins.
Each pilot/comparison build requires a unique `KEYPRINT_RUN_ID` build argument
and the appropriate `--no-cache-filter` setting described in `INTEGRATIONS.md`.
The exported run ID prevents confusing a cached result with the declared attempt.

The September 19 source-identity rerun returned ten texts (five complete pairs,
731 tokens matching their condition-specific selection journals), then reached
the original 420-second deadline while generating the JSON pair. All three
callback contract checks passed, but the full inference run did not. Its partial
outputs and failed exit status are retained. The complete-suite deadline is now
900 seconds, followed by forced termination after 15 seconds if needed. This
changes the engineering timeout only, not the prompts, token limits or sampling.

The next complete run returned all twelve texts and 620 tokens, passed both
process exits and all three callback contract checks, and passed the independent
receipt verifier. All outputs reached EOS. Three outputs failed mechanical
screens, but those screens undercount semantic problems: the ordinary email
turned a rescheduling request into two meetings, the ordinary negation changed
the required action, the marked science answer introduced an ocean-reflection
error, and both Spanish/French pairs were inadequate. Both JSON outputs matched
the requested values. These are small independent samples, not an estimate of
watermark-induced quality loss. No output-quality acceptance follows.

`.github/workflows/sglang-inference.yml` runs the same six pairs on an ARM CPU
runner when relevant adapter or test-runner code changes, or on manual dispatch.
It downloads the pinned public model before generation; inference itself runs
without network access. The CI key is generated per attempt and is never an
account credential. No GPU, hosted model API or publishing secret is needed.
The workflow retains all available public comparison texts, including failures,
but never uploads keys or private token journals. A green engineering check does
not approve semantic quality, detector calibration or production serving.

After exporting a local `comparison-receipts` run, verify it with:

```sh
PYTHONPATH=src python tools/check_sglang_results.py private-sglang-results \
  --run-id <the-unique-KEYPRINT_RUN_ID-used-for-this-attempt>
```

The verifier requires both process exits to be zero, the declared run ID, the
exact six-case suite, all twelve ordinary/marked outputs, the pinned runtime
source identity, and exact returned-token matches to the private journals. It
also checks that each displayed condition matches its generation journal.
Missing, partial, cached or internally inconsistent results fail closed. The
replay uses only Python's standard library and the checked-out Keyprint source.

## September 17 local findings

- Transformers/SmolLM2: 12 generated texts plus two real local provider-object
  rewrites and one actual OpenAI HTTP generation. Runtime checks passed, but
  Spanish truncated, constrained email/French missed required details and JSON
  failed. Both provider rewrites lost the instruction prohibiting publication
  before approval. Lexical checks did not catch that semantic failure.
- MLX/Qwen3-8B: 12 generated texts plus two local provider-object rewrites.
  All paired responses completed. JSON matched exactly and the reviewed negation
  pair preserved the backup condition. Both French outputs changed `09:30` to
  `09h30`, failing exact-literal preservation. The provider rewrite examples
  preserved the approval condition in this small screen; no broad acceptance.
- vLLM/SmolLM2: 12 generated texts, 1,020 final tokens matched selection journals.
  Output review found factual errors, weak multilingual text, invented email
  details and JSON failures. Final-token compatibility does not repair those
  quality problems.

These observations block a general-purpose rewrite-quality claim. Failures also
occurred in ordinary samples, so this screen does not isolate watermark-induced
quality loss. Matching/other-key bit fractions are uncalibrated diagnostics.
Launch readiness still requires broader inference lifecycle coverage, held-out
quality evaluation and calibrated detection.
