# What works, what is open

Keyprint's first community release is a research toolkit. The goal is to make
watermark mechanisms easy to inspect, compare and improve together. Production
qualification remains a separate milestone, not an implication of publication.

## Available to build on

- Python generation and ordinary/marked comparison APIs for scoped local models.
- Prefilled playground, custom prompts, exact text/token traces and recorded exports.
- Optional local OpenAI, Anthropic, Ollama-client, LangChain and routing recipes.
  Each has specific limits in [COMPATIBILITY.md](COMPATIBILITY.md).
- Retained experimental findings, test fixtures and explicit failure behavior.
- MIT-licensed code, by Vincent Wu (Cveinnt), with third-party notices retained.

## Five contribution tracks

| Track | A useful first contribution | Evidence required |
| --- | --- | --- |
| Detection | Reproduce a published diagnostic and propose a separate scorer | Full cohort, fixed rule, fresh confirmation before a detection claim; no threshold fishing |
| Quality | Add a small public multilingual source/task fixture | Source facts, ordinary/marked outputs, original failures, review rubric and model identity |
| Demos | Build a viewer from `Comparison.to_dict()` or an exported gallery | Reproducible example, unchanged text, honest diagnostic labels, keyboard/mobile checks |
| Integrations | Qualify one specific model/runtime pair | Actual inference, tokenizer/EOS identity, exact text reconciliation and failure/cancellation checks |
| Reliability | Improve installation or resource/lifecycle behavior | Minimal reproducer, focused tests, measured environment, retained failure receipts |

## Findings contributors should know

The latest completed paced-candidate diagnostic cleared its fixed cutoff for
1/64 marked outputs and 0/64 ordinary outputs. It did not establish useful
detection. Strict full-task review passed 16/64 marked and 25/64 ordinary outputs;
both arms contained factual failures. These are scoped development findings,
not a claim that every possible method or base model fails.

The newer balanced study remains incomplete: 71 complete outputs, three
interruptions, 54 unstarted attempts. It has not earned quality or detector
acceptance. See [detection](evidence/paced-key-rank-2026-09-29/README.md),
[quality](evidence/paced-quality-2026-09-29/README.md), and
[balanced study](evidence/balanced-chain-execution-2026-09-29/README.md).

The preview uses the SDK source validated at `bcd2803`. Subsequent unvalidated
durable-server work is excluded. Local server replay lasts only for its process;
do not interpret a restart as authorization to repeat an uncertain request.

## Release scope

No requirement to support every framework before sharing this toolkit.
No blanket hosted GPT/Claude watermarking claim: their public clients do not
expose Keyprint's required native sampling hooks. No production authorship
decisions, guaranteed meaning preservation or “Anthropic reverse-engineered” claim.
Contributions can move those research questions forward without pretending they
are settled. [Detailed evidence history](RELEASE_GOAL.md) remains available.
