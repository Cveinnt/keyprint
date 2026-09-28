# PydanticAI with the local Keyprint worker

The actual-inference integration check passes for **PydanticAI 2.51.0**, its
OpenAI 3.20.0 client and the pinned Qwen3-8B MLX backend. This closes the previously
untested one-turn PydanticAI text/native-JSON path. It does not qualify agent tool
loops, system prompts, conversation history, streaming, hosted GPT/Claude or other
models/backends.

## Verified behavior

- Four ordinary generations and four marked client generations: English,
  French, Spanish and native typed JSON. Every generation reaches EOS.
- `run_sync` works for English; async `run` works for the other cases. Repeating
  each identical request with its original idempotency key returns equal output
  and usage with no additional model generation.
- Three text outputs match private reports verbatim. Typed JSON matches parsed
  values, including Maya, Vincent, `Tuesday 09:30` and
  `publish_without_approval=false`. Typed output is a parsed object; the test does
  not claim that reserializing it preserves the original JSON whitespace.
- Conflicting reuse of a request ID returns 409. System instructions, message
  history and streaming return 400 before model work. Implicit tool-output mode
  is rejected by the declared client profile. Native JSON must be explicit.
- Exactly four marked requests and four HTTP journals exist. The server shuts
  down cleanly. No hosted-provider requests, external tracing or automatic
  retries are enabled.
- A fresh wheel is installed in an isolated environment. All **90 package files**
  match source, wheel and installation. All **216 committed tokens** across the
  eight generations reconstruct native output bytes and match usage and journal
  commit counts. Journal hash chains pass consistency checks.
- The recipe contract and relevant server/MLX structured-output tests pass:
  **44 tests**. No core dependency, SDK sampler or server behavior changed.

See [readable paired samples](SAMPLES.md), [all results](results.json),
[independent receipt/install audit](audit.json), [registered plan](plan.json)
and [checksums](sha256.json). Languages and the tested approval/backup conditions
are retained. Both French answers use indirect phrasing about asking Maya; this
is not full style approval or a general semantic-preservation guarantee. No
detector calibration or scientific clue is accepted by this integration check.

## Recipe and reproduction

The optional [recipe](../../examples/pydantic_ai_local.py) uses the documented
[custom OpenAI client](https://pydantic.dev/docs/ai/models/openai/) and explicit
[NativeOutput](https://pydantic.dev/docs/ai/core-concepts/output/) interfaces.
PydanticAI remains an application dependency, outside Keyprint's package extras.
The tested native-JSON route needs the server's `structured` extra.

From an environment containing the installed Keyprint wheel, MLX/server/structured
extras and `pydantic-ai-slim[openai]==2.51.0`:

```sh
python tools/validate_pydantic_ai.py --model /path/to/pinned-qwen-snapshot \
  --wheel /path/to/keyprint-0.1.0a1-py3-none-any.whl --output private-pydantic-check
python tools/audit_pydantic_ai.py private-pydantic-check \
  --wheel /path/to/keyprint-0.1.0a1-py3-none-any.whl
```

Use a fresh output directory and keep its key/journals private. Only `public/`
is intended for export. Dependency versions and wheel/source hashes are recorded.

## Retained harness failure

The first run generated four ordinary outputs and completed an English client
request with equal text replay, then its usage assertion called `result.usage()`.
PydanticAI 2.51.0 exposes `usage` as a property, so that check raised `TypeError`.
The [failed run](failed-usage-check/results.json), plan and original script/recipe
are retained. The corrected assertion is covered by a contract test, and the
complete integration was rerun in a new directory. This was a validation-harness
failure, not a silently discarded generation or a claimed successful first run.
