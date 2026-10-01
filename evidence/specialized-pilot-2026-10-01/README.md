# Actual inference: code and specialized text

All **32 planned outputs** completed at EOS on the released SDK's default
Qwen3-8B/MLX sampling path. The dataset contains eight new synthetic tasks, two
fresh fixed keys and paired ordinary/marked generation. Cases, executable checks,
keys and source identities were committed before inference. No model download,
response replacement, post-generation rewriting or favorable rerun was used.

| Domain | Ordinary strict passes | Marked strict passes |
| --- | ---: | ---: |
| Python behavior | 4/4 | 3/4 |
| SQLite query results | 4/4 | 4/4 |
| Exact typed JSON, including French values | 4/4 | 4/4 |
| Exact CSV fields | 2/4 | 2/4 |
| **Total** | **14/16** | **13/16** |

Thirteen pairs pass both arms, one passes only ordinary, and two fail both.
Nine pairs are exactly identical; these remain in the results. Two reused tasks
per domain and two keys are a small development pilot, not independent proof of
general code quality, negligible watermark harm or all-clue conformance.

## What passed and what failed

- Python checks exercise approval conjunction/negation and an exact shipping-fee
  threshold, including expedited orders above the free-shipping boundary.
  One marked answer wraps otherwise correct-looking code in forbidden Markdown.
  The original parser rejects that answer; no fences were stripped to create a pass.
- SQL checks exercise unknown approval values, exact Boolean-like integers and
  aggregation before filtering. Every output returned the expected rows.
- JSON preserves status booleans, names, deadlines, roles, and the French Friday
  dispatch versus unconfirmed delivery distinction. Every output passed exact
  value/type checks. These were plain generation prompts, not grammar-constrained
  outputs or a post-processing repair.
- All four maintenance CSV outputs change `Kumo` to `kumo`. One marked output
  additionally changes the exact timezone identifier `Asia/Tokyo` to `asia/tokyo`.
  Those fail the machine-readable field contract. The separate role/status CSV
  task passes throughout. Casing failures are distinct from translation or a
  changed approval condition, but can still break downstream consumers.

[All prompts and original outputs](SAMPLES.md) · [Results and receipt hashes](results.json)
· [Frozen plan](plan.json) · [MIT dataset](../../tools/specialized_cases.json)

## Validation boundaries

The checker never executes arbitrary generated Python: a small AST interpreter
supports the declared Boolean/conditional subset. Unsupported syntax is reported
separately from incorrect test results and cannot count as a strict pass.
SQLite uses an in-memory fixture, denies writes/attachment/unknown functions,
and limits query work. JSON rejects duplicate keys and wrong types. CSV compares
parsed cells exactly. Reference solutions pass; ten regression tests cover unsafe
code/SQL, lost negation, boundary mistakes, NULL semantics, duplicate keys,
timezone/role changes and forbidden Markdown.

The full receipt audit reconciles 1,220 committed tokens, output text, assigned
conditions, usage counts, complete journal chains and original oracle results.
It is not independent model-forward replay. Matching-key bit fractions remain
uncalibrated diagnostics; this pilot supplies **no detector-power qualification**.
Low-entropy code/structured outputs especially need that separate evaluation.

This is an author-created, MIT synthetic fixture set, not an externally collected
benchmark or a training-held-out claim. No domain-wide semantic guarantee follows.
The historical A02/A03/A18 gates stay open. The result adds narrow functional
evidence and concrete first-use failure cases; it does not erase prior prose failures.

## Memory and retained interruptions

The first start rejected macOS AppleDouble `._model.safetensors` metadata in the
external-drive cache before inference. A clean symlink view of the same five
pinned model assets passed the unchanged asset verifier. Original files remained
untouched. This exposes a loader usability issue with that cache layout.

After the first 16 outputs, the guard stopped on system memory pressure while
loading the second key, before any second-key generation began. Cleanup was
verified; pressure returned to normal. An explicitly recorded fresh-worker
continuation ran only the unstarted 16 attempts, preserving the first 16 byte for
byte. Its peak footprint was 5.36 GiB under an 8 GiB guard; cleanup succeeded.
The stopped worker peaked at 6.25 GiB. These are model-worker footprints, not SDK
package memory. No unrelated app was stopped. Both interruptions and the final
successful continuation remain in [resource receipts](resources.json).

## Reproduce

`tools/validate_specialized.py prepare` freezes a new output directory and fresh
keys; `run` executes it under the watchdog/MLX wrapper documented in
[MEMORY_SAFETY.md](../../MEMORY_SAFETY.md). Use the pinned existing Qwen3-8B snapshot
`545dc4251c05440727734bcd94334791f6ab0192`. Do not reuse this study's output
directory or quietly retry an interrupted generation. `continue_specialized.py`
is specific to the retained pre-generation interruption and refuses other states.
`audit_specialized.py` requires every planned result. This adds research tools
and evidence only; SDK sampling code is unchanged.
