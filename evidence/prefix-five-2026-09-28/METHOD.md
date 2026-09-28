# Five-layer development comparison

The previous half-mixture diluted all thirty tournament layers after sampling
weights were formed. This candidate instead executes only the first five
tournament layers, retaining the original thirty-layer PRF domain for comparison.
The sampling law and runtime identity are separately versioned. It is not an
SDK profile, a change to the default, or a claim about Anthropic's implementation.

In exact arithmetic a tournament layer multiplies a normalized weight by at most
two. Five fixed layers limit this multiplicative amplification compared with
thirty. This motivates the experiment; it does not guarantee factual correctness,
semantic preservation or greater entropy at every model head. Numeric floors,
excluded tokens and protected probabilities retain the reference conventions.
Repeated contexts and ordinary generation retain their original behavior.

The research session first invokes the reference validation/state machinery,
then replaces its active probabilities and numeric counters with the five-layer
calculation. This redundant reference work makes runtime unsuitable for serving
latency claims. The dedicated research caller does not bypass SDK admission or
change package source.

## Frozen generation study

Every existing key slot, including the previously failing key, is used. Per key:
two French explanations, one approval email, and one long explanation in English
(even slots) or Spanish (odd slots). Each instance has ordinary, current-reference
and five-layer arms, giving 48 attempts. Order rotates; no retries, substitutions,
key selection or threshold search. The pinned Qwen3-8B MLX model, temperature 0.7,
top-k 100, prompts and rubrics match the previous development screen. Short/long
caps are 192/768 tokens. Exact plans and source hashes precede inference.

All texts are rated against the fixed rubrics before condition/key metadata is
joined. This is assistant review with hidden metadata, not independent human
review or full blinding: earlier samples, candidate design and prompts were known.
Every failure and cap remains in the denominator. No human acceptance is inferred.

## Detection and controls

The reference retains its thirty-layer weighted score. The candidate uses the
exact fair-binomial upper tail over its first five layer bits. Both detectors
score every output and both keys (matching and next-slot). The per-key cutoff
is 0.005, a nominal two-key family target of 0.01. Never combine the detectors
by taking whichever passes. Cross-score results expose any difference caused
by changing the score rather than the generator.

The frozen development rule rejects promotion on fewer long-text matching hits,
fewer task passes, or more ordinary/wrong-key hits than the reference. Even a
pass would justify only a larger fresh study. Neither ideal fair-bit arithmetic
nor this small control set establishes a fixed-key deployment false-positive rate.

A separate diagnostic applies the five-layer detector to all 500 previously
opened public-corpus responses and the same two historical keys. Text hashes
must match their earlier records. There is no new selection or tuning. This
is an additional rejection screen, not fresh calibration; source texts and
keys are not redistributed. Any IID confidence bound remains conditional on
assumptions that do not establish fixed-key or cross-domain transfer.

## Reproduce

With the pinned model and existing private September 24 key study:

```bash
PYTHONPATH=src python tools/compare_prefix_candidate.py \
  --model /path/to/545dc4251c05440727734bcd94334791f6ab0192 \
  --prior /path/to/multikey-prose-2026-09-24 \
  --output /new/private/prefix-study
```

Freeze all rubric ratings and the hashes of the plan, review inputs and ratings
before running `tools/summarize_prefix_candidate.py`. It verifies registered
sources, identities, native token bytes, commits, model calls and journal chains.
Hashes establish consistency, not independent authenticity. New keys/model
revisions constitute a new experiment. The optional MLX research environment
also needs NumPy and SciPy; these are not new core SDK dependencies.
