# Output quality contract

Preserve the requested language, facts, actors, negation, conditions, dates and
relationships. Do not invent facts, approvals or completed actions. An unrequested
translation is a failed output. A correct-looking sentence or watermark signal
does not override these requirements.

## Current enforcement

All post-generation rewrite entry points reject before inference. Their literal
checks are historical diagnostic helpers, not delivery authorization. The native
sampler operates during generation; the SDK must not rewrite or translate its
rendered response afterward. Existing journal-to-output checks verify the sampled
bytes. Model generation itself remains capable of instruction and factual errors.

The inference comparison harness assigns every attempted output either `blocked`
(runtime failure, missing measurements or failed mechanical checks) or
`unreviewed`. It approves no outputs for delivery. Literal passes cannot become a
semantic pass. JSON and HTML show the same verdict. Regression cases cover
unrequested translation, lost negation, approval replaced by acknowledgement,
swapped actors, swapped times and invented approval. These tests verify the
reporting boundary; they do not implement a semantic detector.

## Evidence required before qualification

Freeze prompts, language and meaning obligations, model/tokenizer/runtime,
sampling settings and acceptance rule before generating ordinary and marked
samples. Retain every result and error, including failures in either condition.
Review both responses against the task, not merely against each other: two
responses can share the same mistake. Separate baseline model failures from
watermark-attributable harm. Do not select only favorable examples or rerun a
failure until it passes.

Any translation, changed condition or invented fact in a proposed supported
workflow blocks its qualification until addressed and independently rechecked.
The fixed multilingual decision screen is not prose-quality evidence. The older
A02 harm bound exceeds its registered target, A03 remains unestablished, and the
serving-cost gate remains open. Historical scoped acceptances do not transfer to
a new model, integration, prompt or SDK configuration.

No finite test suite can promise error-free LLM output. Launch claims must state
the exact verified scope. Disabling unsafe rewriting is a safeguard, not closure
of output quality or a replacement for the user's intended capability.
