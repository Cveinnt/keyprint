"""Independent full-sort oracle and C/FFI failure boundaries."""
from concurrent.futures import ThreadPoolExecutor
import ctypes
import struct

import numpy as np
import pytest

native_module = pytest.importorskip('keyprint_native')


@pytest.fixture(scope='module')
def native(): return native_module.NativePRF()


@pytest.mark.parametrize('size', [1, 100, 1024, 151669])
@pytest.mark.parametrize('top_k', [1, 100, 512])
def test_binary32_order_matches_float64_oracle(native, size, top_k):
    k = min(size, top_k)
    rng = np.random.default_rng(size + k)
    bits = rng.integers(0, 2**32, size=size, dtype=np.uint32)
    # Replace nonfinite patterns without changing the finite bit-pattern sample.
    bits[(bits & 0x7f800000) == 0x7f800000] = 0
    random = bits.view(np.float32)
    tiny = np.nextafter(np.float32(0), np.float32(1))
    edge = np.resize(np.array([-np.finfo(np.float32).max, np.finfo(np.float32).max,
                              -tiny, tiny, -0., 0.], dtype=np.float32), size)
    for scores in (random, edge, np.rint(rng.normal(size=size)).astype(np.float32),
                   np.zeros(size, dtype=np.float32), np.arange(size, dtype=np.float32),
                   -np.arange(size, dtype=np.float32)):
        raw = scores.astype('<f4').tobytes()
        expected = tuple(np.lexsort((np.arange(size), -scores.astype(np.float64)))[:k])
        assert native.select_indices(raw, k) == expected
        assert raw == scores.astype('<f4').tobytes()


@pytest.mark.parametrize('scores,k', [(b'', 1), (b'abc', 1), (bytes(4), 0),
    (bytes(4), True), (bytes(4), 2), (bytes(513*4), 513), (bytes(151670*4), 1),
    (bytearray(4), 1), (bytes(4), 1.0)])
def test_python_boundary_rejects_before_call(monkeypatch, native, scores, k):
    monkeypatch.setattr(native, '_select', lambda *a: pytest.fail('C call reached'))
    with pytest.raises(ValueError): native.select_indices(scores, k)


@pytest.mark.parametrize('word', [0x7f800000, 0xff800000, 0x7fc00001, 0x7f800001, 0xffc00001])
def test_nonfinite_late_input_leaves_c_output_untouched(native, word):
    data = bytes(2048*4) + struct.pack('<I', word)
    output = (ctypes.c_uint32*104)(*([0xdeadbeef]*104))
    before = bytes(output)
    assert native._select(data, len(data), 100, output, 400) == 0
    assert bytes(output) == before
    with pytest.raises(RuntimeError, match='no partial result'):
        native.select_indices(data, 100)


@pytest.mark.parametrize('size,k,outsize', [(0, 1, 4), (3, 1, 4), (4, 0, 0),
    (4, 2, 8), (4, 1, 3), (4, 1, 8), (513*4, 513, 513*4), (151670*4, 1, 4)])
def test_c_size_bounds_before_read_or_write(native, size, k, outsize):
    output = (ctypes.c_uint32*520)(*([0xdeadbeef]*520))
    before = bytes(output)
    # Tiny input is safe only if invalid bounds reject before any score reads.
    assert native._select(bytes(4), size, k, output, outsize) == 0
    assert bytes(output) == before


def test_nulls_and_partial_failure_are_not_results(native, monkeypatch):
    output = (ctypes.c_uint32*1)(123)
    assert native._select(None, 4, 1, output, 4) == 0
    assert output[0] == 123
    assert native._select(bytes(4), 4, 1, None, 4) == 0
    monkeypatch.setattr(native, '_select', lambda *a: 0)
    with pytest.raises(RuntimeError, match='no partial result'):
        native.select_indices(bytes(4), 1)


def test_concurrent_calls_do_not_share_heap_or_output(native):
    def run(seed):
        scores = np.random.default_rng(seed).normal(size=4096).astype(np.float32)
        expected = tuple(np.lexsort((np.arange(len(scores)), -scores.astype(np.float64)))[:100])
        assert native.select_indices(scores.astype('<f4').tobytes(), 100) == expected
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(run, range(64)))
