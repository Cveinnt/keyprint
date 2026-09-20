# Keyprint native accelerator (unreleased)

Private packaging candidate for the optional native HMAC batch helper. This is
not a standalone watermark detector or a published SDK integration.

Prebuilt wheels contain the native library and statically linked OpenSSL;
installing one requires no compiler, Homebrew or separate OpenSSL installation.
The current builder targets Apple Silicon macOS only. Other platforms remain
unqualified. The wheel tag is taken from the built library's minimum OS version,
not guessed from the build host or Python interpreter.

The loader checks bundled file hashes before loading the library. Those checks
catch accidental corruption; they are not a signature or proof of authenticity.
It exposes full digests and a build identity, uses no network and keeps no keyed
context across calls. Model inference and full-caller acceptance are separate.

## Install a reviewed local wheel

The package is not on PyPI. Given the exact reviewed wheel from a private build:

```sh
python -m pip install ./keyprint_native-0.1.0a1-py3-none-macosx_11_0_arm64.whl
```

Install the core SDK and its MLX dependencies separately. With the pinned model
assets already available, opt in explicitly:

```python
from keyprint import Keyprint

key = Keyprint.new_key()
model = Keyprint.from_mlx(
    "models/qwen3-8b-4bit",
    key=key,
    execution="experimental-native",
)
result = model.generate("Explain why the sky is blue.")
print(result.text)
```

Retain the same private key when inspecting results later; the snippet creates a
new key for this process. Missing or corrupted accelerator files fail before
model loading. There is no silent fallback to another execution.

The September 20 wheel bundles OpenSSL 3.6.4 and has no external crypto-library
dependency. A clean environment containing only this package passes full-digest
and installed-file integrity checks. Its macOS 11 ARM64 build target is verified
from the binary and static archive; actual execution was tested on macOS 26.2,
not every older macOS release. The wheel is approximately 2.1 MiB.

Complete Qwen/MLX caller comparison retains 12 pairs, 24 outputs and 988 committed
tokens with exact reference/native text, sampling-record and token parity. Both
OpenAI and Anthropic local clients pass cancellation after a committed token,
terminal replay, new-request reuse and graceful shutdown. These checks do not
qualify hosted GPT/Claude sampling, other model bindings, detection, quality or
production serving. See [performance evidence](../PERFORMANCE.md).

Explicit source builds require `KEYPRINT_OPENSSL_ROOT` pointing to a static
OpenSSL 3 installation and an Apple compiler. Source builds never download
OpenSSL. The builder rejects unexpected dynamic dependencies. Packaging and
verification receipts identify the pinned source archive and compiler options.

MIT project code by Vincent Wu (Cveinnt). Bundled OpenSSL is Apache-2.0; its
license is included. Neither package has been published by this work.
