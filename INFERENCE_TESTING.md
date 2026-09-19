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
