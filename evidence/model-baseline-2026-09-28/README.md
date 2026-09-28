# A newer ordinary model still invents facts

Ran eight actual ordinary responses using **Qwen3.5-9B, MLX 4-bit**, revision
`8b2b98c00a6b4d291155e4890773ca8f769aee53`. All eight reached EOS and retained
the requested language. Two pass the strict factual and full-task review;
accepting both pre-flagged conservative judgments gives four factual passes
and three full-task passes. No watermark was applied.

| Measure | Ordinary upstream model |
| --- | ---: |
| Execution | 8/8 |
| Requested language | 8/8 |
| Strict facts and no unsupported additions | 2/8 |
| Content with both conservative judgments accepted | 4/8 |
| Requested length and three paragraphs | 5/8 |
| Strict full task | 2/8 |
| Full task with both conservative judgments accepted | 3/8 |

Failures include an invented Tuesday, missing contact details, invented shipping
validation/loading procedures, guarantees beyond the customer's recorded request
and a new condition on when documentation may proceed. The strongest responses
preserve the Japanese maintenance facts and French pricing/support facts.
Conservative judgments cover ambiguous French capacity/annual wording and
English team/safe-staging language. All labels and their reasons are retained.

This isolates an important limitation: factual failures can occur without a
watermark. It neither establishes watermark-caused harm nor excuses such harm.
These are the same four opened prompts as the previous long-form study, with a
different model, tokenizer and upstream sampler. Do not pool their results or
claim a controlled improvement over the previous SDK run. Assistant reviewers
knew the model and tasks; only seed/repetition metadata was hidden before the
labels were committed. This is not independent human acceptance or a new
all-or-nothing launch gate.

## Concrete SDK integration work remains

The current Keyprint MLX adapter is pinned to Qwen3-8B. Qwen3.5 has a 248,320-token
head, beyond the current portable binding/filter limits, and an NFC normalizer,
which the existing binding rejects. Generation rendering and literal detector
replay must be audited separately: normalizing arbitrary input before replay
cannot be silently treated as unchanged text.

The nested model config records EOS 248044, while the actual MLX tokenizer loader
resolves `<|im_end|>` to **248046**. All eight outputs terminate at 248046. The
first independent audit used the nested field and correctly failed reconciliation;
the corrected audit reconstructs the upstream loader's actual policy, recording
both values. No generation, source prompt, seed, output or rating was changed or
rerun. [Audit corrections](audit-attempts.md) and
[binding preflight](binding-preflight.json) preserve the mismatch. A future
adapter must bind the actual stop policy rather than copying the nested field.

No current SDK limits were loosened, no frozen sampler files changed and no
compatibility or detector acceptance transfers to this model. A separately
identified wider-vocabulary/native MLX binding needs exact token-byte rendering,
Unicode replay checks, explicit stop IDs and sampling-law verification before
paired marked-generation testing. Base-model quality and watermark-added harm
remain distinct questions.

## Reproduction and receipts

- [Every output and source prompt](SAMPLES.md).
- [Metadata-hidden review page](review.html), blank human answers with explicit export.
- [Frozen plan](plan.json), [rating commitment](rating-commitment.json),
  [frozen ratings](frozen-ratings.json), [results](results.json).
- [Model selection and resource limits](selection.md).

Two attempts per task; predetermined unique seeds; temperature 0.7, top-k 100,
1,024-token cap; thinking explicitly disabled. No retries, replacements,
post-generation repairs, output cherry-picking or watermark keys. All 2,391
sampled tokens reconcile with retained upstream events and native decoding;
2,407 actual model-forward calls were counted. This is upstream token/decoding
evidence, not the Keyprint journal or calibrated detector path. All 13 downloaded
model files passed Hugging Face cache verification before inference.

Seven new baseline receipt tests pass, including EOS, caps, native whitespace,
terminal-event consistency and token-count mismatch; 35 combined baseline/long/short
study checks pass locally. No CI, deployment, package publication or posting.

```sh
hf download mlx-community/Qwen3.5-9B-4bit \
  --revision 8b2b98c00a6b4d291155e4890773ca8f769aee53
python tools/validate_model_baseline.py \
  --model /path/to/8b2b98c00a6b4d291155e4890773ca8f769aee53 \
  --output /new/private/baseline
# Review public/blind-review.json and freeze ratings plus commitment first.
python tools/audit_model_baseline.py /new/private/baseline \
  --model /path/to/8b2b98c00a6b4d291155e4890773ca8f769aee53
```

Model references: [Qwen's model card](https://huggingface.co/Qwen/Qwen3.5-9B)
and [MLX conversion](https://huggingface.co/mlx-community/Qwen3.5-9B-4bit).
Published benchmark claims motivate evaluation; they do not qualify Keyprint.
