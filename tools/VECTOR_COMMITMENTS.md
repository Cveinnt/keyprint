# Lossless vector commitments: experimental native v2

The core SDK now includes `keyprint.experimental.vector_commitment` for explicit
experimental native v2 execution. Default and frozen reference receipt formats
keep their dense SHA-256 hashes. The independent `tools/vector_commitment.py`
prototype remains a test oracle and reproduces the historical experiments below.
Neither package has been released, and this format does not close a release gate.

Native report schema is `keyprint.experimental-native-report.v2`, and its runtime
version is `keyprint-mlx-native-experimental-v2`. Sampling records contain
`base_probability_commitment` and `prepared_probability_commitment`; private
filter attempts contain `filtered_logits_commitment`. Each object declares
`encoding`, `kind`, `width` and `sha256`. Encoding is `keyprint.binary64-vector.v1`.
Probability width is the complete 151,669-token mapped vocabulary; filtered
logits retain the complete 151,936-entry head. Old dense hash field names are
rejected by the new report contract, and legacy reporting rejects these objects.
No reader should infer an encoding from a 64-character digest alone.

Reports explicitly say `vector_commitments_resolved: false`. Their validator
checks structure and the existing random-draw, token and completion invariants;
it does not claim to resolve a vector from a hash. The codec and report validator
sources are part of the runtime identity. Saved legacy reports remain legacy
artifacts; they are not rewritten or passed off as new-format evidence.

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

Reproduce the helper screen with an installed core/native environment:

```sh
python -m pytest -q tests/test_vector_commitment.py tests/test_vector_commitment_audit.py
python tools/benchmark_vector_commitment.py --output NEW_HELPER_SCREEN
```

The historical probe needs the preceding native v1 environment. It hashes
original bytes unchanged and additionally observes packets.
Its module-local instrumentation is explicit. Each output directory must be
new; all attempts stay recorded. Packets and owner keys are private, outside
version control. The repaired probe has not received another inference run;
the recorded actual-vector evidence comes from the initial captured outputs.

## Integrated parity and qualification

`tools/commitment_parity.py` explicitly instruments native v2 in a test process.
It independently encodes each actual vector using the retained prototype,
checks exact reconstruction and compares the packet hash to the SDK commitment.
The offline auditor resolves those retained packets back to their original dense
bytes before comparing hashes with the reference reports. It also checks packet
order, count, source identity, journals, text, random draws and consumed work.
No digest is relabeled without resolving its bytes. Instrumentation is restored
on exit and is absent from serving measurements.

```sh
python tools/validate_fast_caller.py --model PINNED_QWEN \
  --execution experimental-native --output NEW_PRIVATE_PARITY
python tools/audit_fast_caller.py --run NEW_PRIVATE_PARITY
```

At the vector-commitment integration stage, the installed wheel passed 1,079 checks without skips, including 64 codec
cases against both independent and SDK implementations, sampler/failure parity,
channel isolation and malformed v2 reports. All 83 package files match source
and wheel. Twelve actual caller pairs match across 988 tokens after vector
resolution; both client lifecycle checks and eleven structured requests pass.

That stage's unchanged uninstrumented v2 study measured marked/engine seconds per token
at 1.093868 (upper 1.100527), auditing 2,972 tokens across 72 measured outputs and
three warmups. The preceding native v1 result was 1.137564 (upper 1.146169).
Independent draws, output lengths and background applications prevent treating
the difference as an isolated causal estimate. The complete-path 1.05 target
still fails. No cost, quality, detection, production or launch gate closes.
Later filter changes and their failed full-path cost study are recorded in
[current performance evidence](../PERFORMANCE.md#full-gap-range-check).
Hosted CI stays disabled; this work runs locally.
