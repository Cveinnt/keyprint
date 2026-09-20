# Integration readiness

Reviewed September 16, 2026. This is an engineering audit and implementation plan,
not a claim that the planned adapters are available.

### Current private development branch

`clean-keyprint-sdk` builds the unpublished `keyprint==0.1.0a1` package and
public Python namespace. The reference engine is namespaced with a recorded
derivative manifest and behavior comparisons, without changing the released
bundle. A Transformers CPU float32 adapter now generates text with
SmolLM2-135M-Instruct at revision `12fd25f77366fa6b3b4b768ec3050bf629380bac`.
Its ByteLevel profile is explicitly experimental and receives no reference
scientific acceptances. A local OpenAI-client endpoint and experimental local
rewriter are implemented. A separate vLLM CPU source-level pilot completed two
batched generations. Broad framework and hosted-provider product readiness
remain open; details below.

The repository is temporarily private while this work is reviewed. The old
PyPI research package and website remain public. No new launch date is set.

### September 20 optional native MLX execution

An unpublished `keyprint-native` wheel provides explicit
`execution="experimental-native"` for the pinned Qwen/MLX binding on Apple
Silicon macOS. It contains its own static OpenSSL 3.6.4 dependency and needs no
compiler or separate crypto installation when using the wheel. Binary and
wrapper hashes are part of its distinct runtime identity; reference scientific
acceptances do not transfer. Default execution remains unchanged.

Installed-wheel regression checks pass 700 tests, with twelve native tests
initially skipped and then covered by a separate 25/25 boundary suite. Seven
JavaScript tests pass. Full-caller comparison passes twelve reference/native
pairs and reconciles all 988 committed tokens. Actual OpenAI and Anthropic
clients each cancel after one committed token, replay the terminal response
without another generation, then reuse the worker for a successful 32-token
request. These are local text-protocol tests, not hosted provider access.

The native wheel's macOS 11 target is a verified build property. Execution is
tested on macOS 26.2 ARM64 only; Linux, Intel Macs and older macOS execution are
not qualified. See [native install instructions](native/README.md) and
[performance scope](PERFORMANCE.md). CI stays disabled and publication stays held.

### September 20 local SmolLM3 extension

The Transformers adapter now also runs HuggingFaceTB/SmolLM3-3B at revision
`a07cc9a04f16550a088caea529712d1d335b0ac1`, using CPU float32, temperature 0.7,
top-k 100 and its non-thinking chat template. Its eight non-special added ASCII
tokens have verified ByteLevel rendering; ambiguous added-token forms still
fail before model loading. This does not qualify other tokenizer families or
SmolLM3 under the native SGLang/vLLM adapters.

A fresh installed wheel passed 523 Python and seven JavaScript tests. Actual
inference completed all twelve paired-case outputs, two local provider-object
rewrites and two HTTP client requests. Independent reconciliation matched 544
generated tokens, source/assets identities, journal chains, text and usage.
Both OpenAI and Anthropic local clients replayed without another generation.
All sixteen requests reached EOS. No hosted provider calls were made.

This fixes integration blockers, not quality acceptance: both JSON responses
included prohibited Markdown fences; the ordinary email invented a reason;
the ordinary French response changed a deadline into an availability question.
The marked science response used one sentence instead of two. Both negation
samples and the two synthetic rewrites retained their respective conditions. All
outputs and a post-hoc unblinded assistant review are retained. Neither the
small sample nor mechanical screens establish semantic equivalence, detection
calibration or acceptable overhead. CI remains manually disabled.

## What works now

The published rc3 SDK supports supplied NumPy logits and a separately supplied
MLX model with the frozen Qwen tokenizer and 151936-column model head. The
GitHub MLX example completed ordinary and marked responses using the pinned
Qwen3-8B-4bit checkpoint. Installed-wheel tests passed on Linux/macOS and Python
3.12/3.13. These checks do not validate other model families or servers.

## Three different meanings of compatibility

| Integration | Feasible route | Current Keyprint status |
| --- | --- | --- |
| OpenAI Python client against a self-hosted server | Standard client with a custom base URL; watermarking happens in our inference server | Real HTTP request using OpenAI 3.14.1, pinned Qwen/MLX and idempotency replay passed; one user message, no streaming/tools |
| OpenAI-hosted GPT models | Public API has model-dependent sampling controls, including logit bias and limited returned log probabilities; it has no arbitrary per-token custom sampler callback | Current candidate cannot be inserted into hosted generation |
| Anthropic-hosted Claude | Messages API generates the response remotely; no custom pre-sampling logits hook is documented | Current candidate cannot be inserted into hosted generation |
| SGLang | Custom logits processor and request parameters, integrated inside the serving stack | Pinned ARM CPU source build: six ordinary/marked SmolLM2 pairs, 620 matching returned tokens; NUMA workaround required, output quality and production lifecycle unvalidated |
| vLLM | Stateful batched logits-processor extension; native watermarking is also documented upstream | Experimental CPU 0.29.0+cpu: latest two batched SmolLM2 requests returned 125 tokens matching private selection journals; production server lifecycle and broader models unvalidated |
| Hugging Face Transformers | Keyprint owns a single-response CPU sampling loop and KV cache | SmolLM2 real generation tested on the private branch; broader model and quality coverage missing |
| MLX | Public `run_response` caller and pinned local example | One exact model/tokenizer tested |

An OpenAI-compatible endpoint is not an OpenAI-hosted model. Reading a Claude
response into Python is not watermarking it. SDK interfaces must make this
distinction explicit, including in examples and error messages.

See [provider usage](PROVIDERS.md). The completed-response helpers parse actual
OpenAI and Anthropic SDK object types, reject unsupported response modes and
run an explicit local rewrite. They return both texts and lexical checks. No
hosted model call was made in these tests. A rewrite can pass lexical checks
while reversing meaning; it always requires review and has no detector claim.

### Experimental vLLM CPU pilot

`keyprint.experimental.vllm.KeyprintLogitsProcessor` runs only with the pinned
CPU version, float32 logits and a validated ByteLevel binding. It owns filtering
and exact integer sampling, then passes a one-token mask to vLLM. Framework
temperature must be 1, top-p 1 and top-k disabled; penalties, seeded sampling,
logprobs, constraints, stop strings and speculation are rejected. Prefix
validation prevents silent resampling. Journals distinguish tentative selections
from subsequently confirmed prefixes; compare the final response independently.

The pilot ran inside official ARM64 image
`vllm/vllm-openai-cpu@sha256:527ec4e8188f2ad480aca5863ab3b7e7c39cfda84f6c0bbb06525363a3eb5a0f`.
It used mounted source with that image's NumPy 2.3.5, SciPy 1.18.1 and tokenizers
0.23.2, rather than the distribution's research pins. Consequently **there is
no supported `keyprint[vllm]` installation extra yet**. The immutable image is
the reproduction environment. `tools/validate_vllm.py` requires read-only model
assets at `/model`, a private key file and a private trace directory configured
through `KEYPRINT_KEY_FILE` and `KEYPRINT_TRACE_DIR`. Model revision is the
SmolLM2 checkpoint above. The initial two responses reached the 64-token cap.
The final source run returned 125 tokens total: one EOS completion and one
token-capped response. All token IDs matched the selection journals. Two repeat
runs lost workers under a 3 GiB container limit; the successful final run used
4 GiB, two CPU threads and a 256 MiB KV cache, with zero recorded OOM events.
This validates integration and token-path alignment, not answer quality or overhead.

`tools/test_vllm_contract.py` exercises the real upstream batch-state API with
small logits fixtures. Engine cancellation, network serving, streaming, GPU
precision, cache reuse and a second tokenizer family still need validation.
The adapter is excluded from the normal CLI and carries no reference evidence.

### Experimental SGLang ARM CPU pilot

`keyprint.experimental.sglang.KeyprintLogitsProcessor` binds CPU source revision
`13d593b6cf885c5c4d50eea88c82b9e28cf5941e` (`sglang-cpu 0.5.20.dev791+g13d593b6c`).
It shares the exact sampler with the vLLM pilot, while keeping each request's
state across SGLang processor reconstruction. Prefix mismatches fail instead
of silently reinitializing the watermark. Tests cover reconstruction, changed
keys, unsupported settings and journal cleanup when a request is collected.

The September 19 rebuild of that same checkout produced version label
`0.5.21.dev69+g13d593b6c`. The earlier string-only guard killed the worker before
any output was returned. The adapter now accepts those two observed labels only
when five installed source files match SHA-256 fingerprints from the pinned
commit: custom processor, sampling parameters, sampler and server arguments.
Unknown labels, missing files and changed source fail closed. The comparison
runner performs this check before allocating workers and retains its identity.
These fingerprints cover the Python sampling contract, not all upstream code
or compiled kernels; actual runtime evidence remains separate.

The complete September 19 comparison run returned all twelve texts across six
cases, with 620 final token IDs exactly matching condition-specific selection
journals. All three contract checks passed. Both JSON outputs parsed with the
requested values; multilingual, factual and email instruction failures remain.
This qualifies the recorded CPU callback path only, not semantic quality or
the production serving lifecycle. Earlier failed and partial runs are retained.

The unmodified ARM build crashed in NUMA initialization on the local Docker VM.
The recorded pilot skips optional NUMA memory binding while retaining CPU thread
binding. This is a **modified runtime**, not an unmodified upstream support pass.
`tools/sglang-arm.Dockerfile` preserves the workaround and exports run receipts
without needing to unpack the full image. It uses named build contexts
`keyprint_source` (this checkout) and `model_assets` (the pinned SmolLM2 files),
plus BuildKit secret `keyprint_key` (a private 32-byte file). The model run has no
network access. Read `exit-code.txt` and `contract-exit-code.txt`: exporting
receipts successfully does not itself mean the contained run passed.

Pass a fresh `--build-arg KEYPRINT_RUN_ID=<unique-attempt-id>` for every pilot or
comparison attempt, and use `--no-cache-filter pilot` or
`--no-cache-filter comparisons` respectively. BuildKit secret contents do not
invalidate cached `RUN` results; changing only a key file must not be mistaken
for fresh inference. The exported `run-id.txt` must match the declared attempt.
The runtime compilation remains cacheable. The `comparison-receipts` target
runs all six ordinary/marked cases and exports results without loading a second
copy of the complete runtime image.

Completed requests are retired on the next sampler callback using the pinned
host's completion state. This closes their journals and detaches finalizers
without waiting for Python's cyclic garbage collector. The 32-request bound
applies to unfinished requests; a retired request cannot resume sampling.
The last idle batch can remain until garbage collection or worker exit. This
does not establish cancellation, crash recovery or long-running service safety.

Two 64-token marked responses completed; all 128 returned IDs matched the
durable selection journals. Both reached the token cap. The environment used
NumPy 2.3.5, tokenizers 0.22.2 and torch 2.12.0+cpu, separately from distribution
pins. Full resolved versions are recorded in the private run's runtime lock.
Rebuilding from source can resolve newer dependencies; it needs a fresh check.
There is no supported `keyprint[sglang]` extra or GPU claim yet.

Only trusted offline execution is tested. Overlap, prefix caching, speculation,
logprobs and constrained sampling are rejected. Arbitrary serialized processors
must never be accepted on a public endpoint. Cancellation, long-running service,
detector controls, quality and comparative overhead remain open. Verify retained
outputs with `tools/check_native_receipts.py`; it checks journal chains and exact
token paths, not detection accuracy or the full mathematical sampling law.

### Native completion text and token limits

For these non-streaming pilots, run returned text and IDs through
`keyprint.experimental.completion.finalize_completion` using the same
`ByteLevelBinding` as the processor:

```python
from keyprint.experimental.completion import finalize_completion

result = finalize_completion(
    binding,
    token_ids=output_ids,
    text=host_text,
    finish_reason=finish_reason,  # The actual host value: "stop" or "length".
    max_tokens=requested_cap,
)
print(result.text)
```

The helper checks that EOS is terminal, a length completion reaches the exact
requested cap, and text agrees with the returned token bytes. Unknown finish
reasons and unrelated text changes raise errors. A capped first token may be
only part of a character, so valid visible text can be empty. The helper keeps
an incomplete UTF-8 tail in `result.pending_utf8` and retains the unmodified
framework string in `result.host_text`. It accepts either an omitted partial
tail or one replacement character for that tail; only the latter is removed.
It never draws another token or repairs malformed interior bytes or an
incomplete EOS completion. A genuine sampled replacement character is retained.

This explicit helper does not change text emitted directly by framework servers
or qualify streaming. Match returned IDs against the private sampling journal
separately. `tools/validate_native_cases.py` now uses this helper and retains raw
host responses before validation. The `utf8-receipts` Docker target exercises
six fixed multilingual/emoji prompts at six short caps in both conditions; use
a fresh run ID and `--no-cache-filter utf8`, then check the exported exit status
and every attempt. Its short identical paths are checked as a complete multiset
against journals, not presented as uniquely matched request identities.

## Hosted-provider options worth evaluating

1. Generate with GPT/Claude, then use a local marked rewriter. This can ingest
   either provider's visible text, but changes wording and adds a second model
   pass. Facts, citations, code, multilingual meaning, latency and detection
   need new evaluations. Never silently rewrite tool calls, JSON or thinking
   blocks. Return the original and transformed text with an explicit mode.
2. Select from multiple provider-generated candidates using a keyed rule. This
   needs a separately designed and evaluated black-box watermark. Extra API
   calls, refusals, deterministic responses and selection bias are material.
3. Token-by-token hosted requests with bias are not equivalent to the audited
   sampler. Limited returned probabilities cannot reconstruct every token's
   probability, and repeated chat calls can change the context and cost.

None is a drop-in adapter for the frozen reconstruction. Do not copy its 22
scoped acceptances onto a different algorithm or postprocessing mode.

## Portable-engine work required

The preserved reference implementation hard-codes vocabulary dimensions, byte decoding,
EOS and channel token IDs, normalization policy, and a particular tokenizer.
Accepting a different model name is insufficient. The portable ByteLevel binding
now records tokenizer bytes, special tokens, EOS and decoding identity. Other
tokenizer families and generated channels still need explicit bindings, fixtures
and real-model evaluation.

The exact-integer sampler operates on the post-filter weights. A conventional
logits processor that reweights before the framework's top-k/temperature/sample
steps can change the law. Match filter ordering and the commit boundary; a
successful hook invocation does not prove equivalence. Fail unsupported beam,
speculative, grammar-constrained or streaming modes explicitly until tested.

Prioritize SGLang, then Transformers and vLLM. For each adapter require:

- Pinned framework version and at least two distinct tokenizer/model families.
- Real generation and matching-key replay; wrong-key and unmarked controls.
- Concurrent requests, changing batch order, cancellation and cache reuse.
- Stable prefix handling; EOS, Unicode, normalization and output-channel tests.
- Streaming/final equivalence where streaming is supported.
- Quality and end-to-end overhead measurements, including failure cases.
- A standard OpenAI-client request to the self-hosted endpoint.

Keep secret watermark keys on the server. SGLang's custom processor source
serializes callables with dill; do not expose arbitrary client-supplied
serialized Python on an untrusted public endpoint. Register approved processors
server-side and select them by a constrained identifier.

## Naming migration (original findings, now implemented on private branch)

The next-release branch adds `keyprint demo`, `keyprint doctor`, and
`keyprint verify`, with readable output and opt-in `--json`. The old CLI remains
available for existing scripts. This command change does not rename the PyPI
distribution or fix the Python import namespace.

Target public surface is `pip install keyprint`, a `keyprint` command, and a
documented `keyprint` Python package. Neither `keyprint` nor `keyprint-sdk` had a
public PyPI JSON record at audit time; 404 does not guarantee registration is
permitted or reserve a name. Publication and trusted-publisher configuration
must be verified before advertising the new install command.

The frozen legacy core already imports a package named `keyprint` and checks
its exact file location. Do not hide that conflict with `sys.modules` swaps or
pretend a renamed import preserves provenance. Move internals into a private
namespace with explicit provenance and regression comparisons, while preserving
the released research artifact. New code identities require new manifests.

## Primary sources

- [SGLang custom processor source](https://github.com/sgl-project/sglang/blob/869674b3a72ea8de4de63e3e302b1052e6b73392/python/sglang/srt/sampling/custom_logit_processor.py)
- [SGLang OpenAI request schemas](https://github.com/sgl-project/sglang/blob/869674b3a72ea8de4de63e3e302b1052e6b73392/python/sglang/srt/entrypoints/openai/protocol.py)
- [vLLM custom processors](https://docs.vllm.ai/en/latest/features/custom_logitsprocs/)
- [vLLM watermarking](https://docs.vllm.ai/en/latest/features/watermarking/)
- [OpenAI Chat Completions API](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create)
- [Claude Messages API](https://platform.claude.com/docs/en/api/http/beta/messages/create)
- [Transformers generation utilities](https://huggingface.co/docs/transformers/en/internal/generation_utils)
