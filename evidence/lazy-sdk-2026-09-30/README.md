# Lazy reference allocation: local validation

Every Keyprint constructor previously built the pinned Qwen reference tokenizer
and scorer, including instances whose backend supplies its own tokenization.
Keys, integrity and sampling parameters now validate immediately; the unchanged
reference engine initializes on first reference use, retaining its thread owner.
MLX reference/fast/native wrappers still initialize their required reference.
Transformers, llama.cpp and experimental wide MLX avoid the unrelated allocation.
No sampling, model weights, text transformation or detection threshold changed.

## Verified

- 119 focused CLI, llama.cpp boundary and lazy-reference tests pass in 1.40 seconds.
  Guarded peak: 138,805,872 bytes (132.38 MiB), cleanup verified. These use small
  native-boundary fixtures, not loaded models, and include the eight previously
  unexecuted CLI cleanup cases.
- 19 further public API tests pass in isolated processes. Three optional-native
  wheel tests skip because that wheel is absent in this environment.
- Full existing parity grid: two conditions, eight seeds, sixteen steps per path.
  All 256 tokens and rendered carriers match the preserved implementation exactly.
  Each preserved/current arm ran separately: 32 successful workers, maximum
  sampled footprint 318.53 MiB. Supplied-logit fixtures, not real inference.
- Current wheel builds and installs into a new offline core-only environment
  outside the checkout on macOS arm64/Python 3.13.13. Dependency check, isolated
  import, doctor, verify, model-free demo and help pass; seven web assets match.
  No optional model framework installed. Full pipeline: 295.38 MiB peak, 18.64
  seconds. Wheel and all 78 Python source hashes appear in `validation.json`.
- Isolated physical-footprint point samples: Python baseline 9.33 MiB; after
  import 12.92 MiB; twenty unbound instances 38.69 MiB. Closing them leaves the
  measured footprint unchanged. No reference tokenizer or model loaded.
  These are point samples, not a peak, leak soak or inference overhead estimate.

## Retained failures and limits

Before the fix, combined tests and the isolated llama.cpp suite crossed the
512 MiB cutoff (sampled overshoot: 534.45 and 552.03 MiB). A single ordinary
generation fixture passed at 350.92 MiB. The combined reference parity test also
crossed that cutoff; its complete grid subsequently passed in separate workers.
All terminated process groups were cleaned up. No budget increase was used.

An isolated cancellation test failed on both frozen and current SDKs: patching
shared `secrets.randbits` affected dependency initialization before inference.
It now replaces only `keyprint.api.secrets`, retaining assertions for one model
call, one committed token and journal ordering. The corrected test passes in a
fresh process. Both failures and the passing receipt remain recorded.

Workers ran sequentially with 512 MiB sampled guards, pressure/disk checks and
verified cleanup. Docker and unrelated apps were untouched. No model loaded.
Earlier approximately 5–6 GiB study footprints include local 9B inference;
they are not blanket SDK memory requirements. Frozen research keeps its separate
SDK snapshot. No semantic-quality or calibrated-detection gap closed here.
Public launch, registry publication and deployment remain held.

External storage had approximately 1.7 TiB free; internal storage approximately
2.8 GiB. The external disk adds storage, not physical RAM. No files moved or removed.
Scripts are execution receipts: use new output directories when reproducing them.
Package resolution was offline; network traffic was not separately audited.
