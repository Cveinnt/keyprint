# Public multilingual fact-preservation fixtures

This fixture-only contribution addresses [issue #4](https://github.com/Cveinnt/keyprint/issues/4).
It contains no model outputs, inference results or quality acceptance. The three
original synthetic prompts concern fictional people and a fictional workshop;
they contain no customer material. They are contributed under the repository's
[MIT license](../LICENSE).

[`tools/multilingual_fact_cases.json`](../tools/multilingual_fact_cases.json)
contains the same six source obligations in Thai, Indonesian and German. Thai
and Indonesian extend the languages in the existing fact-fidelity fixtures.
German provides a same-scenario comparison in an already represented language. These are
authored test inputs, not translations or repairs of generated output. The
English `facts` entries are a reviewer aid; the actual prompt is in the requested
language. No existing case, frozen rubric or evidence is overwritten.

## Freeze before any generation

Record the exact repository revision, the raw fixture SHA-256 and the harness's
canonical case hash before a run. Declare the model, tokenizer, runtime and
their exact revisions, sampling settings, key allocation, attempt schedule and
review method. Retain this rubric unchanged. If a prompt needs correction after
outputs are seen, give the corrected cohort a new identity and keep the old one.

Each case is compatible with `load_cases` in
[`tools/validate_compatibility.py`](../tools/validate_compatibility.py). Additional
`language` and `facts` fields are review metadata. The 768-token cap bounds an
attempt; it is not evidence that every tokenizer will finish. There is no
whitespace word limit: it would not be a comparable Thai length measure. The
single-paragraph requirement is reviewed separately from factual correctness.

## Prospective review rubric

Review each untouched output against its own source prompt, before consulting
watermark diagnostics or comparing it with the other condition. Assign each
of the six facts `pass`, `fail`, or `ambiguous`, with a quotation from the
output and a short reason. Compound facts require every component:

1. Total 18; reserved 11; available 7. Reservations are not attendance or payment.
2. Saturday at 10:30 is proposed, not confirmed. No calendar date is supplied.
3. Mira is responsible for checking room readiness; that role is not Narin's.
4. Narin is responsible for sending the confirmation notice to participants;
   that role is not Mira's.
5. Sending requires Mira's prior readiness confirmation. Do not weaken "only
   after" to an optional preference, or treat silence/reservations as approval.
6. Readiness confirmation and dispatch of the confirmation notice have not
   happened. Neither a future plan nor responsibility implies completion.

Record separate verdicts for requested language, unsupported claims, exact
required literals and one-paragraph format. Names, numerals and the time may
appear as specified without failing language retention. Equivalent idiomatic
wording is acceptable; literal matching does not establish factual correctness.
For example, "reserved" must not become "paid", and "not yet confirmed" must
not become "cancelled". Do not require a particular sentence order.

No readiness confirmation does not mean the room is unready. An unconfirmed
schedule does not mean the event is cancelled. The prerequisite does not
establish that confirmation will occur or that dispatch is automatic.

Mark a clear omission, contradiction, weakened condition, incorrect completion
claim or changed responsibility `fail`. Mark
genuinely unclear coverage, pronoun references, modal strength or completion
status `ambiguous`, and retain the original text and explanation. Do not resolve them
using the other arm's output or a watermark score. A strict full-task pass
requires all six facts and all separate verdicts to pass; ambiguity is not a
strict pass. Any descriptive sensitivity count accepting ambiguities must be
reported separately with the original flags preserved. Record reviewer identity
or method and whether a fluent human reviewed the judgments. AI review is not
independent human acceptance.

## Later inference is a separate contribution

Follow [the inference protocol](../INFERENCE_TESTING.md) and
[output-quality contract](../OUTPUT_QUALITY.md). Keep both ordinary and marked
texts, model/runtime identities, completions, errors and unstarted/interrupted
attempts. Never translate, rewrite, repair, selectively omit or retry a failed
output to obtain a pass. Runtime failures and truncations remain failures, not
empty successful responses. Keep literal screens, semantic review and signal
diagnostics separate.

No inference command is run or required by this fixture-only change. A later,
explicitly scoped run may pass this file as `--cases` to the existing harness;
it must use a declared local model and a fresh output directory. Do not download
models, enable hosted CI or make hosted provider calls merely to review these
fixtures. A future comparison must distinguish ordinary model failures from
marked failures; this three-case, single-topic fixture cannot establish a
language-wide effect, causation, noninferiority or general quality acceptance.

## Model-free checks

From the repository root:

```sh
python -m unittest discover -s tests -p 'test_multilingual_fact_cases.py' -v
git diff --check
```

The tests check JSON shape, stable case/language identity, matched source facts,
explicit literals, bounds, and separation from earlier fixtures. They do not
certify translation accuracy, runtime compatibility or output quality. The
fixture wording and rubric were AI-assisted and source-compared; a fluent human
review of each language remains a useful additional check before inference.
