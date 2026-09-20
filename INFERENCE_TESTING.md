# Actual inference compatibility checks

Run the same six prompts through ordinary and marked sampling, retaining both
texts, completion reasons, timings and matching/other-key diagnostics. Cases
cover science, a constrained email, negation, Spanish, French and strict JSON.
Independent samples are compared descriptively; differences are not a causal
estimate of watermark quality loss.

## Reproduce locally

Hosted `SDK checks` and `SGLang actual inference` workflows were manually disabled
at the user's request on September 20 after repeated pre-execution billing-limit
failures. Keep them disabled until the user requests otherwise. Run checks
locally; do not repeatedly dispatch GitHub jobs as an availability probe.

Against a freshly installed wheel with the applicable extras:

```sh
python -m pytest
node --test tests/test_reader.cjs tests/test_playground_ui.cjs
keyprint doctor
```

These commands do not replace the actual inference checks below or establish
coverage for another operating system, model or backend.

The local September 20 ARM Docker SGLang repetition completed twelve outputs and
661 journal-matching tokens, with four mechanical screening failures retained.
Its verifier also checks the recorded Keyprint native-adapter and sampling-source
identities against the installed package. A token-path match alone cannot qualify
a different source version. Passing this check does not approve generated prose
or authenticate a receipt supplied by an untrusted party.

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

For the larger SmolLM3-3B checkpoint, download the explicit revision in README.md
and substitute `--model models/smollm3` in the same command. The portable adapter
passes `enable_thinking=False` and records that template policy in each identity.
The template includes the current date; exact prompt token IDs are retained in
each private journal, so a later-day repeat is not an identical-prompt replay.
SmolLM3's eight ordinary added tokens have exact ASCII/ByteLevel byte agreement.
Added tokens containing whitespace, non-ASCII characters or matching
transformations remain rejected; they require a separate verified binding.
The original SmolLM2 prompt tokens are unchanged by the non-thinking option.

### Structured output and typed clients

Install the `structured` extra with Transformers, server and clients, then run:

```sh
python tools/validate_structured.py --model models/smollm3 \
  --output private-structured-run
```

This retains unconstrained controls, ordinary/marked schema-constrained JSON,
Unicode and negation examples, a deliberate one-token cap, and real OpenAI and
Anthropic typed parsing requests. Schema types constrain format; requested field
values are checked separately. Both clients must replay without another model
attempt. A separate matcher replays every retained grammar mask and committed
token, checking its hash, membership, final text, usage and validation receipt.
The original control failures remain visible in `public/comparison.html`.
The whole run fails on an inference, lifecycle, typed parsing or audit error;
mechanical/semantic output review remains separate from that engineering result.
This test is local; hosted CI stays disabled.

The HTTP check runs actual OpenAI and Anthropic SDK requests through a TCP socket
to one local model worker. It verifies generated text against private reports,
exact idempotency replay, one generation per client and rejection of unsupported
options. Anthropic uses the local Messages text subset with its standard API-key
header; it does not call hosted Claude. Earlier retained runs predate this endpoint.
Provider-object rewrite checks construct real SDK response objects from a fixed
synthetic document and run the local model. They are not hosted GPT/Claude calls.

Run just the two real client requests and their receipt/replay checks:

```sh
python tools/validate_clients.py --backend transformers --model models/smollm2 \
  --output private-client-run
```

For the pinned Qwen model use `--backend mlx`, its model path, and optionally
`--execution experimental-fast`. For actual Anthropic-client cancellation,
run `tools/validate_cancellation.py` with `--protocol anthropic` and the same
backend/model/output flags. The deliberately controlled cancellation boundary
tests retained work and worker reuse, not natural cancellation latency.

Test a token limit reached inside a UTF-8 character on real Transformers
inference, independently for ordinary and marked sampling:

```sh
python tools/validate_portable_utf8.py --model models/smollm2 \
  --condition ordinary --output private-utf8-ordinary
python tools/validate_portable_utf8.py --model models/smollm2 \
  --condition marked --output private-utf8-marked
```

The declared multilingual prompt/seed order selects the first partial-character
prefix. It then reruns the actual model at that exact cap and compares every
model-head hash, weight hash, random draw and commit with the original prefix.
No extra token is generated to complete the character. Every search output stays
in the report; failure to find or reproduce a partial prefix exits unsuccessfully.
This is a boundary regression, not an output-quality or failure-rate estimate.

Compare the repaired default MLX caller against the frozen research caller using
independently executed real model forwards, identical fixture draws, and fresh
response caches:

```sh
python tools/validate_fast_caller.py --execution reference \
  --model models/qwen3-8b-4bit --output private-bounded-reference-parity
python tools/audit_fast_caller.py --run private-bounded-reference-parity
```

The historical tool name is retained; `--execution reference` selects the new
bounded reference caller as the candidate. The comparison side always uses the
archived strict caller. Omitting this flag tests experimental-fast instead.
The audit binds sources, prompts, conditions, token caps, journal chains, model
sampling records and generated text. Results establish scoped parity only.

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

The same tool now accepts `weighted-*` rows from the frozen two-key power study.
It resolves each original prompt by source identity, checks its hash, checks the
selected key commitment, and records the source public-row hashes. For example:

```sh
python tools/analyze_mlx_capacity.py --study private-weighted-power \
  --model models/qwen3-8b-4bit --output private-short-capacity \
  --cases weighted-04-marked weighted-08-marked weighted-09-marked \
          weighted-10-marked weighted-11-marked
```

The September 19 replay selected all five marked answers in the earlier
100–400-word control range, including the one detected answer. All 1,590 model
steps matched original raw logits, prepared distributions, tokens and saved
draws, with zero new random draws. Every original result remains unchanged:

| Case | Words | Original text-only flag | Steps with base max probability >90% | Model-path log2 likelihood ratio |
| --- | ---: | --- | ---: | ---: |
| 04 | 130 | Miss | 76.2% | 27.18 |
| 08 | 112 | Miss | 84.6% | 28.73 |
| 09 | 276 | Miss | 79.9% | 116.14 |
| 10 | 329 | Hit | 70.5% | 169.34 |
| 11 | 237 | Miss | 80.2% | 68.40 |

These positive path ratios show marking evidence that the existing text-only
statistic does not recover in four cases. They require the original prompt,
model and private execution state; they are not five successful text-only
detections, confidence percentages or deployment calibration. Most steps offer
little token choice, while the text score also includes their key-dependent
bit variation. This motivates testing whether a prompt-free estimate of token
predictability can recover signal without an unacceptable false-positive rate.
It does not establish that such an estimator works. Any candidate must be frozen
before fresh power and null evaluation; do not tune thresholds on these cases.

A separate replay of all 24 original outputs found identical generation/text
tokenization in 23. The sole mismatch was a long, already-detected marked
answer (case 07, with 907 shared eligible context/label events out of 912
generation events and 914 literal events). All five short answers matched
exactly, so a tokenization mismatch does not explain these four misses.

## Compare and confirm a text-only score change

### Prompt-free predictability development

```sh
python tools/develop_predictability_filter.py --study private-weighted-power \
  --model models/qwen3-8b-4bit --output private-predictability-development
python tools/audit_predictability_filter.py --development private-predictability-development \
  --study private-weighted-power --model models/qwen3-8b-4bit
python tools/develop_predictability_null.py --development private-predictability-development \
  --source private-corpus/databricks-dolly-15k.jsonl \
  --null-study private-weighted-null --model models/qwen3-8b-4bit \
  --output private-predictability-null
python tools/audit_predictability_null.py --run private-predictability-null \
  --development private-predictability-development \
  --null-study private-weighted-null \
  --source private-corpus/databricks-dolly-15k.jsonl
```

This separate research tool uses the fixed prefix `Continue the text.` and
preceding literal tokens to estimate next-token entropy with pinned Qwen weights.
The fixed sampling filter is temperature 0.7 and top-k 100. It selects only
previously unused canonical contexts with entropy at least 0.5 bits, then applies
the unchanged linear weights and two-key reference rule. Selection cannot use
the original prompt, watermark key, current token probability or observed bits.
All 24 opened paired outputs remain in the report, including misses and errors.
Per-position model-head hashes, token IDs and entropy diagnostics stay private.
Text inference uses fixed batches of 64 positions; only the final batch is padded
and padded future heads are discarded. Its cache is never reused for another
text. A separate audit recomputes every reported score and compares 32-token and
96-token literal prefixes with fresh caches. It requires identical selected
positions and entropy differences below 0.01 bits for shared prefix tokens.

The declaration is saved before model inference. It fixes both the entropy
cutoff and a gate for further development: recover at least one additional
short marked answer, retain overall marked hits, and introduce no ordinary or
wrong-key hits. Passing that gate would justify an ordinary-corpus control
screen, not a production detector. The null command also requires a passed
integrity audit and unchanged candidate-source hashes before loading a model.
It retains all 500 previously opened controls, errors and unavailable results;
it does not choose a threshold from them. This method still requires an 8B model and
additional inference. It neither establishes low-cost detection nor extends
support to other models, frameworks or hosted providers. Source responses are
already opened development data; fresh confirmation remains necessary.
After completion, the separate null auditor reconciles the exact set of 500
source documents and hashes, private position selections, both keys' bit replay,
each reference tail and a doubled FFT grid. It recomputes flags at the unchanged
threshold and checks the one-sided 97.5% IID-only bound against SciPy's beta
quantile, independently of the run's binomial-CDF implementation. Missing,
duplicate, substituted, failed or unavailable controls cannot receive a passing
audit. An integrity pass does not establish independent documents, fresh
validation or an acceptable deployment false-positive rate.

The September 19 null run ended incomplete: 500 attempts, 499 usable controls,
eight flags, and one disk-write failure at source index 5030. The independent
audit rejected it with "Failed or unavailable controls cannot count as negatives".
The reported IID-only bound remains null. Preserve this run and its failed audit;
any recovery must be separately declared and must not overwrite the original
failure or silently count it as a negative.

The separate `tools/audit_partial_predictability_null.py` command can inspect
surviving receipts while retaining every failed attempt and the original failed
audit. It validates source and selection hashes, both keys' scores, doubled
FFT grids and the same-control baseline. It writes `partial-integrity.json`,
never overwrites `integrity.json`, and cannot issue complete-run acceptance or
a confidence bound. It does not rerun model heads or recover missing results.

The September 20 partial replay verified all 998 surviving text/key scores.
Maximum doubled-grid difference was 8.53e-13 or less. On the same 499 controls,
three flags were shared, five were new and two baseline flags disappeared:
eight candidate flags versus five baseline flags. Combined with the opened
power screen, this is a sensitivity/false-flag tradeoff, not an unqualified
improvement. It is not a statistical comparison on fresh held-out data. Keep
the candidate outside the SDK; do not tune the cutoff on these observed controls.

The September 19 opened-data screen improved marked hits from 8/12 to 10/12
and short-answer hits from 1/5 to 3/5, without ordinary or wrong-key hits in the
24 outputs. Cases 04 and 11 were recovered; 08 and 09 still missed. Case 08's
per-key reference tail was 0.0052866, above the unchanged 0.005 cutoff; it remains
a miss. Do not round it into acceptance or relax the threshold after seeing it.

The first version used a shorter final inference batch. A fresh-cache prefix
check found up to 0.1841 bits of entropy variation when batch shape changed,
although that prefix's selected positions were unchanged. That failed audit and
its source snapshots remain retained. Fixed-size batches reproduced the same
10/12 and 3/5 results. All 48 shortened-prefix checks then produced identical
raw model-head hashes, entropy values and selected positions, and all 48
key/text reference scores were independently recomputed. The improvement remains development evidence; it does not
close the detection release gate or establish the false-positive rate.

### Fixed layer-score comparison

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

## Subsequent short-text detector development

The same 24 opened responses have now been used for the following development
comparisons. They are no longer held-out confirmation data for candidate choice.
The five short marked responses contain 100–400 words. All four methods had zero
flags in the paired negative checks (12 ordinary texts under two keys, plus 12
marked texts under the wrong key); those 36 checks do not establish a deployment
false-positive rate.

| Development method | Marked hits | Short marked hits | Decision |
| --- | ---: | ---: | --- |
| Weighted-bit reference | 8/12 | 1/5 | Insufficient short-text power |
| Hard predictability filter | 10/12 | 3/5 | Separate null run adds false flags |
| Full surrogate tournament likelihood | 7/12 | 0/5 | Reject development candidate |
| Model-centered bit betting mixture | 8/12 | 1/5 | Reject development candidate |

The predictability filter's separate 500-control run had eight flags among 499
usable controls and one recording failure. Preserve that incomplete result;
the failed case is neither a negative nor a candidate for silent replacement.

The final row uses `tools/residual_bet.py` and `tools/develop_residual_bet.py`.
For each new canonical context and layer it subtracts the model's predictive
bit average from the observed bit. Five fixed betting stakes are averaged over
the whole document; it never selects the best stake after seeing the text.
The per-key cutoff remains `log(200)`. Excluded, repeated-context and
out-of-support observations contribute no evidence while literal context advances.
This candidate reuses the retained key-blind model heads and makes no new
generation or model-inference calls. It fails the predeclared requirement of
at least 10/12 overall and 4/5 short detections with no negative-check flags.

`tools/audit_residual_bet.py` verifies all source/score hashes, 16,135 retained
head identities, all 48 score aggregations and all 32,270 eligibility decisions.
An independent direct-PRF calculation matches 240 fixed sampled residual events.
This is explicitly a bounded diagnostic audit, not full residual replay or
authorization for null expansion. Ten tests include exhaustive ideal-PRF
normalization over two contexts, repeated-context exclusion, canonical-label
grouping and mixture aggregation.

The ideal random-key argument does not establish fixed-HMAC-key deployment
calibration, finite-precision error guarantees or robustness to key-dependent
text. Probability-aware detection and betting processes have prior research
([AISTATS 2025](https://proceedings.mlr.press/v258/li25d.html),
[e-process framework](https://arxiv.org/abs/2602.14286)); this is an independently
implemented development candidate, not a claimed reproduction or novel method.
No SDK verdict, generation law, detector threshold or release acceptance changes.

These results rule out promoting the tested formulas. They do not rule out all
model-assisted detection. Another candidate needs a distinct mechanistic reason
and a frozen design; any selected method still needs fresh power and null data.

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
