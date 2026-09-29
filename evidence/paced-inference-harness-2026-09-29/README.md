# Guarded paced-candidate inference harness

The candidate now has a separate native-forward runner, streamed token receipts,
sampling replay and blinded review export. **No model inference was performed for
this harness qualification.** All 78 frozen SDK Python sources remain unchanged;
the candidate is not promoted into the SDK. Preparing a study does not close the
quality, detection or serving gaps.

## What is ready

`tools/run_paced_source.py` verifies the existing audited 128-output study, its
sixteen frozen inputs/rubrics, all four original keys, every original prompt
journal, pinned model assets, runtime and SDK. It takes no output-based subset:
the full schedule retains all 128 original case/key/condition combinations and
their order, temperature 0.7, top-k 100 and 768-token cap. Summarization uses the
general purpose; it is not silently routed through proofreading protection.

The separate preflight uses the first original ordinary/marked pair, with eight
tokens each. Full execution requires a completed and audited candidate preflight
whose code, assets, keys, prompts, settings and profile still match. A plan-only
invocation is insufficient. Actual execution requires both the external watchdog
and cache-disabled MLX wrapper. Each output gets a fresh model cache; cleanup
clears it and synchronizes Metal before proceeding.

`tools/paced_inference.py` passes the recorded prompt to a native forward, then
passes only the exact committed token on subsequent forwards. It retains native
head hashes, sparse base weights, exact integer distributions, all valid random
draw attempts and committed token IDs in a flushed journal. EOS and caps are
distinct. Forward, numerical, entropy and decoding failures retain their partial
evidence; no response retry, text repair, rerouting or alternate sample is used.
Abrupt termination leaves incomplete evidence, not a successful result.

User-visible text must match the strict UTF-8 decoding of committed token bytes,
excluding explicit EOS only. Decoder cleanup, normalization or rewriting cannot
silently change those bytes: mismatches retain the rejected decoder output and a
failure. This prevents rendering changes; it does **not** prevent a model from
generating a false fact or a foreign-language word in its sampled tokens.

The output auditor replays the session and recorded random draws, independently
checks exact rational CDF selection and normalized probability bounds, reconciles
EOS/cap counts and source receipts, and checks rendered text against committed
bytes. It shares the sampling kernel and does not replay native model heads.
Correct-key and wrong-key token-path counts remain uncalibrated; they are not
detector decisions. Fresh OS random bits are used during inference. Ordinary and
marked pairs share prompts, not sampling randomness.

Source passages, keys, derived text, prompt IDs and review packets stay private.
The public plan contains hashes and case identifiers. A complete run would export
a shuffled review packet without condition/key/count fields; no ratings have been
made or accepted for this candidate.

## Verified locally

**141 focused checks pass:** the previous 103 mathematical/numerical/session checks
plus 38 harness checks. New tests cover native callback sequencing, EOS/cap/error
retention, unchanged UTF-8 fragments and whitespace, decoder-rewrite rejection,
sampling/commit/text tampering, changed source evidence, original schedule
selection and rejection of stale/incomplete preflight receipts. They use synthetic
native-head callbacks. They are not actual inference or a fresh full-SDK suite.

The latest plan-only invocation verified the full existing assets and 128-attempt
schedule under a 512 MiB watchdog: 77,398,544 bytes (~73.8 MiB) sampled peak,
normal exit and verified cleanup. Earlier planning attempts remain retained.

Development checks caught two integration mistakes before model execution:
closing an already-closed EOS/failure session, and reading prompt IDs outside
the original journal's event envelope. Both were fixed and covered by tests or
the real frozen-input planning invocation.

Recorded [plan](plan.json) and [source/resource validation](validation.json).

## Next execution

Establish sufficient system memory headroom, then run a fresh named preflight:

```sh
PYTHONPATH=src:tools python tools/memory_watchdog.py \
  --output /private/new-preflight-supervisor --limit-gib 10 --timeout 180 -- \
  python tools/mlx_guarded_worker.py tools/run_paced_source.py \
  --root /private/original-source-run --prior /private/original-key-study \
  --model /pinned/local/model --output /private/new-preflight --stage preflight
```

After successful receipt review, use a separate guarded study invocation with
`--stage study --preflight /private/new-preflight` and fresh output directories.
No automatic restart. Retain every failure and cap. Then assess all paired texts,
calibrate detection independently and measure actual serving cost before any SDK
promotion or launch. The original diagnostic replay remains separately incomplete
at 111/128; this candidate does not replace that missing evidence.
