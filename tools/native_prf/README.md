# Native PRF development experiment

This helper tests whether crossing the Python/crypto boundary once per batch
can reduce the measured marking cost. It is **not imported by Keyprint**, not
included as a wheel extension, and does not run during installation. Nothing
here changes the sampling law, SDK runtime identity or release status.

The C helper uses OpenSSL 3's EVP SHA-256 routines and
[`EVP_MD_CTX_copy_ex`](https://docs.openssl.org/3.5/man3/EVP_DigestInit/) for
context reuse. It applies the same fixed 32-byte-key HMAC construction and
address serialization as the existing SHAContext helper. It returns complete
32-byte digests for auditing rather than checking only one output bit.
All keyed contexts are per call; no shared key cache or global state is added.

## Build and check explicitly

Requires an already installed C compiler, OpenSSL 3 headers/libraries and
`pkg-config`. The builder neither downloads dependencies nor modifies a system
installation. Run from the repo with the Python SDK installed:

```sh
python tools/native_prf/build.py --output private-native-prf-build
```

The command prints the library path. Set that exact path below (`.dylib` on
macOS, `.so` on Linux):

```sh
export KEYPRINT_NATIVE_PRF_LIBRARY="$PWD/private-native-prf-build/libkeyprint_prf.dylib"
python -m pytest tests/test_native_prf_candidate.py tests/test_sha_context.py
python tools/benchmark_native_prf.py \
  --library "$KEYPRINT_NATIVE_PRF_LIBRARY" --output private-native-prf-screen
python tools/validate_native_prf_addresses.py \
  --library "$KEYPRINT_NATIVE_PRF_LIBRARY" \
  --profile-run PRIVATE_RETAINED_DOLLY_PROFILE --output private-native-prf-addresses
```

The output directories must be new. Compiled artifacts stay outside source
control. The Python wrapper requires an explicit library path and rejects invalid
keys, dimensions and buffer sizes before calling native code. C independently
validates offsets and lengths before writing. Failed cryptographic calls discard
partial output. Caller-provided raw C pointers must still refer to valid buffers;
this is a development C ABI, not a sandbox.

The standalone memory-boundary harness can also be compiled and run explicitly:

```sh
clang -std=c11 -O1 -g -Wall -Wextra -Werror \
  -fsanitize=address,undefined -fno-omit-frame-pointer \
  tools/native_prf/sanitizer.c -o private-native-prf-build/sanitizer \
  $(pkg-config --cflags --libs openssl)
private-native-prf-build/sanitizer
```

## September 20 local result

macOS ARM64, Python 3.13.13, OpenSSL 3.6.3, Apple clang 21.0.0:

- 2,115,000 timed full-digest comparisons and 10,890 boundary comparisons match
  both the installed Python SHAContext and stdlib HMAC.
- Median native/reference helper-time ratios: 0.2340 (one label), 0.1820 (ten),
  0.1898 (64), 0.1883 (100), 0.1890 (1,000). Timings include context creation,
  ctypes packing and full-digest result copying. They compare these helpers,
  not the exact complete SDK bit-table path or a serving workload.
- All 60 focused tests pass with the explicitly built library, including invalid
  layouts, full digests, binary/Unicode inputs and concurrent independent keys.
- A separate address/undefined-behavior sanitizer harness passes 10,000 calls,
  output canaries and invalid-input checks. OpenSSL itself was not rebuilt with
  sanitizers; this is not a comprehensive memory-safety or cryptographic audit.
- Replaying selected-token addresses from 16 retained real outputs checks 77,490
  additional full digests across 2,647 committed tokens. Original journal chains,
  commit counts and report hashes reconcile. This is address replay, not new
  model inference or complete sampling-distribution replay.

The earlier 1.057928 upper serving-cost ratio still fails the 1.05 screen.
Before SDK integration: choose a supported distribution/binding design, bind
native source/build identity, test complete-caller parity and lifecycle behavior,
then freeze the runtime for a new declared serving study. Linux/Windows delivery,
native-server overhead and production acceptance are unqualified.
