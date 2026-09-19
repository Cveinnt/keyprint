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

## Framework pilots

`tools/validate_native_cases.py` runs the same cases as ordinary/marked pairs in
pinned CPU vLLM or SGLang environments. It compares actual final token IDs with
hash-chained selection journals; text retokenization is not substituted for host
commit verification. Each runtime is separate from the installed-wheel test,
with different dependency versions and explicitly experimental support.

The SGLang Dockerfile exposes a reusable `runtime` target before its historical
pilot and receipt-export stages. It retains the documented ARM NUMA workaround.
Neither framework pilot qualifies GPU serving, streaming, cancellation,
speculation, cache reuse or arbitrary model/tokenizer families.

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
