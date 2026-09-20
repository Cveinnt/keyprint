"""Codec round trips raw bits; it is deliberately outside the SDK."""
from pathlib import Path
import struct
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / 'tools'))
try:
    from vector_commitment import encode_vector, decode_vector, vector_digest, MAX_WIDTH
finally:
    sys.path.pop(0)


def independent_bytes(packet):
    # Separate byte-level reconstruction, without the codec's NumPy record dtype.
    assert packet[:4] == b'KPV1'
    width = int.from_bytes(packet[6:10], 'little')
    if packet[5:6] == b'D': return packet[10:]
    default = bytes(8) if packet[4:5] == b'P' else bytes.fromhex('000000000000f0ff')
    result = bytearray(default * width)
    for offset in range(10, len(packet), 12):
        index = int.from_bytes(packet[offset:offset+4], 'little')
        result[index*8:(index+1)*8] = packet[offset+4:offset+12]
    return bytes(result)


@pytest.mark.parametrize('kind,values,hex_packet', [
    ('probability', [0., 1., 0.], '4b50563150530300000001000000000000000000f03f'),
    ('probability', [1.], '4b505631504401000000000000000000f03f'),
    ('probability', [0.], '4b505631505301000000'),
    ('filtered_logits', [[-np.inf, -0., -np.inf]], '4b5056314c5303000000010000000000000000000080'),
])
def test_cross_language_golden_packets(kind, values, hex_packet):
    a = np.array(values, dtype=np.float64)
    packet = bytes.fromhex(hex_packet)
    assert encode_vector(a, kind) == packet
    assert decode_vector(packet).values.tobytes() == a.tobytes()


@pytest.mark.parametrize('kind', ['probability', 'filtered_logits'])
@pytest.mark.parametrize('width', [1, 2, 3, 32, 1024, 151669, 151936, MAX_WIDTH])
@pytest.mark.parametrize('endian', ['<', '>'])
def test_every_sampled_bit_survives_endianness_and_layout(kind, width, endian):
    rng = np.random.default_rng(width)
    default = 0 if kind == 'probability' else 0xfff0000000000000
    random = rng.integers(0, 2**64, size=width, dtype=np.uint64)
    edges = np.resize(np.array([0, 1, 0x8000000000000000, 0x8000000000000001,
        0x7ff0000000000000, 0xfff0000000000000, 0x7ff8000000000001,
        0x7ff0000000000001, 0xfff800000000002a, 0x7fefffffffffffff], dtype=np.uint64), width)
    sparse = np.full(width, default, dtype=np.uint64)
    sparse[::max(1, width//10)] = edges[::max(1, width//10)]
    for raw in (random, edges, sparse):
        values = raw.astype(endian+'u8').view(endian+'f8')
        # Reversed views exercise negative strides without float conversion.
        for view in (values, values[::-1]):
            original = view.view(endian+'u8').astype('<u8').tobytes()
            shaped = view if kind == 'probability' else view.reshape(1, -1)
            with np.errstate(all='raise'):
                packet = encode_vector(shaped, kind)
                decoded = decode_vector(packet)
            assert decoded.kind == kind
            assert decoded.values.shape == shaped.shape
            assert decoded.values.tobytes() == original
            assert independent_bytes(packet) == original
            assert encode_vector(decoded.values, kind) == packet
            assert view.view(endian+'u8').astype('<u8').tobytes() == original
            with pytest.raises(ValueError): decoded.values.flags.writeable = True


@pytest.mark.parametrize('values,kind', [([], 'probability'),
    (np.zeros(3, dtype=np.float32), 'probability'), (np.zeros(3), 'unknown'),
    (np.zeros((1, 3)), 'probability'), (np.zeros((2, 3)), 'filtered_logits'),
    (np.zeros(3), 'filtered_logits'), (np.zeros(0), 'probability'),
    (np.zeros(MAX_WIDTH+1), 'probability')])
def test_invalid_encoder_inputs(values, kind):
    with pytest.raises((TypeError, ValueError)): encode_vector(values, kind)


def header(kind=b'P', mode=b'S', width=10, magic=b'KPV1'):
    return struct.pack('<4sccI', magic, kind, mode, width)


def entry(index, value=0x3ff0000000000000):
    return struct.pack('<IQ', index, value)


@pytest.mark.parametrize('packet', [b'', b'KPV1', bytearray(header()),
    header(magic=b'KPV2'), header(kind=b'X'), header(mode=b'X'), header(width=0),
    header(width=MAX_WIDTH+1), header(width=2**32-1), header()+b'x',
    header()+entry(10), header()+entry(1)+entry(1), header()+entry(2)+entry(1),
    header()+entry(1, 0), header(width=1)+entry(0),
    header(width=3)+entry(0)+entry(1),
    header(mode=b'D')+bytes(80), header(mode=b'D')+bytes(8),
    header()+entry(1)+b'extra',
])
def test_malformed_packets_rejected_before_output_allocation(packet, monkeypatch):
    monkeypatch.setattr(np, 'full', lambda *a, **k: pytest.fail('output allocated before validation'))
    with pytest.raises(ValueError): decode_vector(packet)


def test_dense_sparse_boundary_and_domain_separation():
    for count, mode in [(0, b'S'), (1, b'S'), (2, b'D'), (3, b'D')]:
        a = np.zeros(3); a[:count] = 1
        assert encode_vector(a, 'probability')[5:6] == mode
    a = np.zeros(10)
    assert vector_digest(a, 'probability') != vector_digest(a.reshape(1, -1), 'filtered_logits')
    a[0] = -0.
    assert vector_digest(a, 'probability') != vector_digest(np.zeros(10), 'probability')
    assert vector_digest(np.zeros(11), 'probability') != vector_digest(np.zeros(10), 'probability')
