# Integration readiness

Reviewed September 16, 2026. This is an engineering audit and implementation plan,
not a claim that the planned adapters are available.

## What works now

The published rc3 SDK supports supplied NumPy logits and a separately supplied
MLX model with the frozen Qwen tokenizer and 151936-column model head. The
GitHub MLX example completed ordinary and marked responses using the pinned
Qwen3-8B-4bit checkpoint. Installed-wheel tests passed on Linux/macOS and Python
3.12/3.13. These checks do not validate other model families or servers.

## Three different meanings of compatibility

| Integration | Feasible route | Current Keyprint status |
| --- | --- | --- |
| OpenAI Python client against a self-hosted server | Standard client with a custom base URL; watermarking happens in our inference server | Not implemented or tested |
| OpenAI-hosted GPT models | Public API has model-dependent sampling controls, including logit bias and limited returned log probabilities; it has no arbitrary per-token custom sampler callback | Current candidate cannot be inserted into hosted generation |
| Anthropic-hosted Claude | Messages API generates the response remotely; no custom pre-sampling logits hook is documented | Current candidate cannot be inserted into hosted generation |
| SGLang | Custom logits processor and request parameters, integrated inside the serving stack | Feasibility confirmed from source; adapter and actual-server validation missing |
| vLLM | Stateful batched logits-processor extension; native watermarking is also documented upstream | Adapter and actual-server validation missing |
| Hugging Face Transformers | Generation logits-processor/custom sampling integration | Adapter and multi-tokenizer validation missing |
| MLX | Public `run_response` caller and pinned local example | One exact model/tokenizer tested |

An OpenAI-compatible endpoint is not an OpenAI-hosted model. Reading a Claude
response into Python is not watermarking it. SDK interfaces must make this
distinction explicit, including in examples and error messages.

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

The implementation currently hard-codes vocabulary dimensions, byte decoding,
EOS and channel token IDs, normalization policy, and a particular tokenizer.
Accepting a different model name is insufficient. Introduce a versioned model
binding for tokenizer bytes, special tokens, decoding, context construction,
and generated channels. Each binding needs fixtures and real-model evaluation.

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

## Naming migration

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
