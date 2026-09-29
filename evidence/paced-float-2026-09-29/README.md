# Certifying rounded output probabilities

A research reference now converts paced proposals to binary64 weights and
certifies the resulting categorical distribution. It uses **exact rational
arithmetic internally**, then rounds output weights once. This is not a fast
floating-point SDK kernel, candidate model inference or production qualification.

The sampler interprets represented weights as exact ratios of their rational
sum. Checking raw weights alone is insufficient: a weight equal to the nominal
lower bound can fall below it after normalization. A regression test demonstrates
that case and requires rejection.

## Reference behavior

- Choose strength before reading labels, retaining the half-to-double total
  bound and per-layer change limit of one eighth of initial probability.
- Reserve a `2^-40` relative rounding margin. A state already within that margin
  of a bound freezes before labels are read. This numerical policy is separately
  identified; it is not bit-for-bit identical to the exact paced oracle.
- Form the proposal with exact fractions; round output weights to binary64.
- Certify exact normalized output probabilities against both budgets and the
  original support. Failure raises an error. No post-label clipping, flooring,
  retries, conditional redraws or alternate-output fallback occurs.
- Preserve original represented weights for identity updates. Admit 1–4,096
  weights and normalized positive base probabilities of at least `2^-900`.
  Smaller probabilities are rejected before output, not silently removed.

The exact pre-rounding update preserves the ideal fair-label conditional mean.
Rounding has a measured error; no exact-unbiasedness claim transfers to these
binary64 outputs or reused deployed keys.

## Measured numerical evidence

All **32 fixed cases and 960 updates** passed. The plan covers four seeds for
uniform, Zipf-like, wide-tail and grouped distributions at sizes 100 and 1,000.
Wide-tail weights span a 600-unit log range; grouped cases share labels between
unequal-weight pairs. Every case is retained. No parameter search, model prompts
or generated text were used.

| Measurement | 100 weights | 1,000 weights |
| --- | ---: | ---: |
| Passed cases | 16/16 | 16/16 |
| Certified updates | 480 | 480 |
| Largest observed rounding total variation | 2.83e-17 | 3.34e-17 |
| Largest grouped ratio relative error | 8.88e-16 | 1.33e-15 |
| Median observed time per layer | 3.04 ms | 31.66 ms |
| Near-boundary freezes | 5 | 0 |

Every support, total-bound and per-layer-bound check uses exact rational values;
the table rounds displayed measurements. Timing includes study-side validation
and measurements and is not a serving benchmark. Thirty layers at the 100-weight
median imply roughly 91 ms of study work per token, so this implementation must
not be presented as a negligible-overhead production replacement.

The run completed under a 256 MiB watchdog with ~14.7 MiB sampled peak footprint
and verified cleanup. It required no model load, GPU work or emulator shutdown.

Fifty-three focused checks pass: the previous 29 oracle/capacity checks plus 24
new numerical-reference checks. They include exhaustive three-layer comparison
with the exact oracle, small measured marginal error, exact-categorical agreement,
larger/grouped supports, invariant violations, unadmitted tiny inputs, identity,
rescaling and predictable freezing. These are not full SDK tests or model-quality
acceptance.

## Next gate

Build and independently compare a faster kernel against this reference without
weakening probability/support checks. Then validate source/EOS/repeated-context
integration and actual paired inference, factual/language outcomes and detector
calibration. The new reference is outside the SDK; its public API and defaults
are unchanged. Original replay remains 111/128; launch remains held.

```sh
PYTHONPATH=src:tools python -m pytest -q tests/test_paced_budget_float_reference.py
python tools/validate_paced_float.py --output /path/to/new-numerical-study
```

Retained [plan](plan.json), [all results](results.json), and
[source/resource validation](validation.json).
