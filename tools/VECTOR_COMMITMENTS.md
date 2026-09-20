# Lossless vector commitments: development prototype

`vector_commitment.py` is outside the SDK. No production receipt format, hash
field, wheel or release gate changes. Existing dense SHA-256 commitments remain
authoritative for the current runtime.

## KPV1 encoding

All integers and IEEE-754 binary64 bit patterns use little-endian order. The
10-byte header contains `KPV1`, one kind byte (`P` for probability, `L` for
filtered logits), one mode byte (`S` for sparse, `D` for dense), and a uint32
width. Width must be between 1 and 1,048,576 inclusive. Probability shape is
`(width,)`; filtered-logit shape is `(1, width)`.

Sparse mode omits only exact default bit patterns: positive zero for `P`,
negative infinity for `L`. Each retained entry is a uint32 index followed by
the original uint64 bit pattern. Indices must be unique, increasing and in
bounds; explicit defaults are forbidden. Use sparse mode only when twelve times
the number of retained entries is less than eight times the width. Otherwise
use dense mode, including ties. Dense mode contains every original bit pattern.
Decoders reject noncanonical alternatives before allocation.

Encoding preserves signed zeros, subnormals and NaN payloads without floating
point arithmetic. It does not validate probability normalization or valid
sampling inputs. The domain tag and width are committed along with the bits.
SHA-256 of this packet is a **new digest contract**, not the existing dense hash.
Neither digest authenticates authorship or provides keyed watermark detection.

## Local evidence, September 20

- 64 codec checks cover exact byte goldens, both byte orders, strides, arbitrary
  binary64 patterns, bounds, malformed inputs and independent reconstruction.
- 240 synthetic encode/hash comparisons preserve all bits. At support 100,
  median candidate/raw-dense time ratios are 0.1227 for probabilities and
  0.1232 for filtered logits. Fully dense vectors are about 8.9% slower.
- One instrumented Qwen run retains twelve outputs, 490 committed tokens and
  1,470 observed vectors. The initial probe failed its reconciliation because
  it expected private filter traces in the exported report. That failed run,
  script, summary and all outputs remain unchanged.
- An offline audit of those same observations passes canonical reconstruction,
  captured file hashes, exported probability hashes and generation journals.
  It checks 1,784,674,080 reconstructed dense bytes against 1,778,700 encoded
  bytes. Filter-trace membership is not checked because that trace is not
  exported. The audit is a separate result, not a passing inference rerun.
- Eleven audit checks reject missing, extra, reordered, altered or mismatched
  observations. No timing of instrumented inference is used as serving evidence.

Reproduce locally with an installed core/native environment:

```sh
python -m pytest -q tests/test_vector_commitment.py tests/test_vector_commitment_audit.py
python tools/benchmark_vector_commitment.py --output NEW_HELPER_SCREEN
python tools/probe_vector_commitments.py --model PINNED_QWEN --output NEW_PRIVATE_PROBE
python tools/audit_vector_commitments.py --run NEW_PRIVATE_PROBE \
  --model PINNED_QWEN --probe-source tools/probe_vector_commitments.py
```

The probe hashes original bytes unchanged and additionally observes packets.
Its module-local instrumentation is explicit. Each output directory must be
new; all attempts stay recorded. Packets and owner keys are private, outside
version control. The repaired probe has not received another inference run;
the recorded actual-vector evidence comes from the initial captured outputs.

## Requirements before SDK integration

Introduce an explicitly versioned receipt schema and encoding metadata; never
reuse old dense-hash field names with new meanings. Keep legacy reading and
failure behavior explicit. Bind codec source into runtime identity. Qualify
probability-level and full-caller parity, journal integrity, provider lifecycle,
structured output and installed wheel behavior. Only then freeze the runtime
for a new uninstrumented end-to-end engine comparison.

The current SDK measurement remains 1.137564 marked/engine seconds per token
(one-sided 95% upper 1.146169), above the 1.05 target. These helper results do
not close cost, quality, detection, production or launch gates. Hosted CI stays
disabled; this work runs locally.
