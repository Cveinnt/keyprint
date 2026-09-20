"""Explicitly opt-in native development helper; never loaded by the SDK."""
import ctypes
from concurrent.futures import ThreadPoolExecutor
import hmac
import os
from pathlib import Path
import sys

import pytest

TOOLS = Path(__file__).parents[1] / 'tools'
sys.path.insert(0, str(TOOLS))
try:
    from native_prf.batch import BatchSHA256
finally:
    sys.path.remove(str(TOOLS))


@pytest.fixture
def native():
    path = os.environ.get('KEYPRINT_NATIVE_PRF_LIBRARY')
    if not path:
        pytest.skip('explicit development library not supplied')
    return BatchSHA256(path)


@pytest.mark.parametrize('key,prefix,suffixes,layers', [
    (bytes(31), b'', [b''], 30), ('x'*32, b'', [b''], 30),
    (bytes(32), 'not bytes', [b''], 30), (bytes(32), bytes(65537), [b''], 30),
    (bytes(32), b'', [], 30), (bytes(32), b'', [bytes(4097)], 30),
    (bytes(32), b'', [b'x']*1001, 30), (bytes(32), b'', [bytes(4096)]*257, 30),
    (bytes(32), b'', ['x'], 30), (bytes(32), b'', [b''], True),
    (bytes(32), b'', [b''], 0), (bytes(32), b'', [b''], 65),
])
def test_python_rejects_invalid_inputs_before_native_call(key, prefix, suffixes, layers):
    instance = object.__new__(BatchSHA256)
    instance.call = lambda *a: pytest.fail('native call reached')
    with pytest.raises(ValueError): instance.digests(key, prefix, suffixes, layers)


@pytest.mark.parametrize('layers', [1, 30, 64])
def test_complete_digests_and_immutable_inputs(native, layers):
    key, prefix = bytes(range(32)), bytes(range(256))
    labels = [b'', b'a\x00b', '你好é'.encode(), bytes(4096)]
    expected = b''.join(hmac.digest(key, prefix+i.to_bytes(4, 'big')+s, 'sha256')
                        for s in labels for i in range(layers))
    assert native.digests(key, prefix, labels, layers) == expected
    assert key == bytes(range(32)) and prefix == bytes(range(256))


def test_concurrent_independent_keys_do_not_mix(native):
    def check(n):
        key = bytes([n])*32
        suffixes = [b'abc', b'\x00\xff', bytes([n])*64]
        expected = b''.join(hmac.digest(key, b'context'+i.to_bytes(4,'big')+s, 'sha256')
                            for s in suffixes for i in range(30))
        assert native.digests(key, b'context', suffixes) == expected
    with ThreadPoolExecutor(max_workers=4) as pool: list(pool.map(check, range(32)))


@pytest.mark.parametrize('offsets,layers,output_len', [
    ([1, 3], 1, 32), ([0, 4], 1, 32), ([0, 3, 2, 3], 1, 96),
    ([0, 3], 0, 32), ([0, 3], 65, 32), ([0, 3], 1, 31),
    ([0, 3], 1, 33), ([0], 1, 32),
])
def test_c_boundary_rejects_invalid_layout_without_writes(native, offsets, layers, output_len):
    sizes = (ctypes.c_size_t * len(offsets))(*offsets)
    output = ctypes.create_string_buffer(b'Z'*128, 128)
    rc = native.call(bytes(32), 32, b'', 0, b'abc', 3, sizes, len(offsets), layers, output, output_len)
    assert rc == 0 and output.raw == b'Z'*128


def test_native_failure_never_returns_partial_digests():
    instance = object.__new__(BatchSHA256)
    instance.call = lambda *a: 0
    with pytest.raises(RuntimeError, match='no partial result accepted'):
        instance.digests(bytes(32), b'', [b'abc'])
