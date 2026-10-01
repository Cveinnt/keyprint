# Existing semantic evidence rechecked

The data was already collected. This pass rechecks all 224 original outputs and
their frozen assistant ratings across two separate cohorts. It does not generate
new text, change ratings, replace failures or pool unlike studies.

| Existing cohort | Ordinary factual passes | Marked factual passes | Language retained |
| --- | ---: | ---: | --- |
| Released SDK profile, 12 synthetic tasks in 7 languages, 4 keys | 42/48 | 36/48 | 48/48 in each arm |
| Experimental Qwen3.5, 16 pinned Dolly source tasks, 4 keys | 33/64 | 23/64 | 64/64 in each arm |

All outputs completed at EOS. Counts match the retained results. The audit binds
plans, blinded output packets and frozen ratings to their original hashes;
verifies one review per planned output and exact review/output text; and recounts
fact and language results independently of the original summary implementation.
It checks stored commitments, not the model forwards or new independent semantic
judgments. [Machine-readable results](results.json).

## What this answers

**Translation:** no requested-language failures were recorded in either cohort.
The multilingual study includes English, French, Spanish, Chinese, German,
Japanese and Portuguese. The Dolly study is English-only. This supports language
retention on those particular tests, not a universal no-translation guarantee.

**Meaning:** strict factual fidelity does not pass reliably. The multilingual
study has nine ordinary-only passes versus three marked-only; Dolly has seventeen
versus seven. Four reused keys and repeated tasks make the observations dependent.
No new causal or noninferiority claim is inferred from these descriptive counts.

**Ambiguity:** keeping the original preflagged sensitivity analysis matters.
Accepting all ambiguous judgments yields 44/48 ordinary versus 43/48 marked on
the multilingual study and 43/64 versus 37/64 on Dolly. These are not confidence
intervals, rerated primary results or proof of unchanged meaning.

## Concrete source-to-output spot checks

These targeted checks revisit already known failures; they are not a new blinded
study. Sources and complete synthetic outputs are in the
[existing sample packet](../fact-fidelity-2026-09-28/SAMPLES.md).

- `ece2f6161353`: Spanish source says dispatch is planned Friday and delivery is
  unconfirmed. Output says “La entrega está programada para el viernes”, assigning
  Friday to delivery. It remains Spanish but changes the operational meaning.
- `ea89ec1e5d82`: Japanese output retains the maintenance times but omits Tokyo
  time, a supplied scheduling detail.
- `08be75201084`: Japanese output adds Friday to October 8, despite receiving no
  weekday or year. The added detail is unsupported.

These are generated responses, not an SDK translation or post-generation rewrite.
The examples demonstrate failures; attributing the change specifically to marking
requires the paired evaluation, not an isolated bad answer.

## Decision and next useful work

A02 remains open. The missing ingredient is not another collection of unrelated
datasets. Existing tests already reveal a fact-preservation problem worth solving.
Keep the released profile separate from experimental candidates; diagnose coverage
and unsupported additions before adding more broad runs. A candidate change needs
a fixed, disjoint confirmation study that checks both fidelity and detection.
Do not tune against this cohort and then reuse it as confirmation.

Reproduce the recount with the original private receipt roots:

```sh
python tools/recheck_semantic_evidence.py \
  --fact-root /path/to/fact-fidelity-2026-09-28 \
  --source-root /path/to/source-grounded-next-2026-09-28/run \
  --output /new/recheck.json
python -m unittest discover -s tests -p test_semantic_recheck.py -v
```

The six model-free regression cases reject missing attempts, duplicate ratings,
altered output text, incomplete fact reviews and nonboolean ratings, and keep
language success separate from factual failure. No SDK sampler, historical
rating, clue acceptance, CI activation or model download changed in this pass.
