"""Lossless binary64 commitments for the experimental native v2 receipt.

KPV1 | kind P/L | mode S/D | uint32 little-endian width. P has shape (width,)
and default +0 bits; L has shape (1,width) and default -infinity bits. Sparse
payloads contain ascending (uint32 index, uint64 bits) records. Dense payloads
contain every uint64 bit pattern. Sparse is canonical iff 12*count < 8*width;
ties use dense mode. Integers and float bit patterns are little-endian.

No float arithmetic, probability validation or empirical acceptance is implied.
Callers must not mutate an input concurrently. Decoded arrays have immutable
backing. Legacy dense hashes must never be relabeled as this format.
"""
from dataclasses import dataclass
import hashlib
import struct

import numpy as np

MAGIC = b'KPV1'
HEADER = struct.Struct('<4sccI')
ENTRY = np.dtype([('index', '<u4'), ('bits', '<u8')], align=False)
MAX_WIDTH = 1 << 20
KINDS = {'probability': (b'P', 0), 'filtered_logits': (b'L', 0xfff0000000000000)}


@dataclass(frozen=True)
class DecodedVector:
    kind: str
    values: np.ndarray


def encode_vector(values, kind):
    if type(kind) is not str or kind not in KINDS:
        raise ValueError('known vector kind required')
    if (not isinstance(values, np.ndarray) or values.dtype.kind != 'f'
            or values.dtype.itemsize != 8):
        raise TypeError('binary64 NumPy array required')
    shape_ok = values.ndim == 1 if kind == 'probability' else values.ndim == 2 and values.shape[0] == 1
    if not shape_ok or not 1 <= values.size <= MAX_WIDTH:
        raise ValueError('bounded vector shape required')
    # Integer byte swapping preserves NaN payloads and signed zero exactly.
    bits = values.view(np.dtype(values.dtype.str.replace('f', 'u'))).reshape(-1).astype('<u8', copy=False)
    tag, default = KINDS[kind]
    different = bits != default
    count = int(np.count_nonzero(different))
    if 12 * count >= 8 * bits.size:
        return HEADER.pack(MAGIC, tag, b'D', bits.size) + bits.tobytes()
    indices = np.flatnonzero(different)
    entries = np.empty(count, dtype=ENTRY)
    entries['index'] = indices
    entries['bits'] = bits[indices]
    return HEADER.pack(MAGIC, tag, b'S', bits.size) + entries.tobytes()


def vector_digest(values, kind):
    return hashlib.sha256(encode_vector(values, kind)).hexdigest()


def decode_vector(packet):
    if type(packet) is not bytes or not HEADER.size <= len(packet) <= HEADER.size + MAX_WIDTH * 8:
        raise ValueError('bounded immutable vector packet required')
    magic, tag, mode, width = HEADER.unpack_from(packet)
    by_tag = {value[0]: (key, value[1]) for key, value in KINDS.items()}
    if magic != MAGIC or tag not in by_tag or mode not in (b'S', b'D') or not 1 <= width <= MAX_WIDTH:
        raise ValueError('invalid vector header')
    kind, default = by_tag[tag]
    size = len(packet) - HEADER.size
    if mode == b'D':
        if size != width * 8:
            raise ValueError('dense payload length mismatch')
        bits = np.frombuffer(packet, dtype='<u8', offset=HEADER.size)
        if int(np.count_nonzero(bits != default)) * 12 < width * 8:
            raise ValueError('noncanonical dense payload')
    else:
        if size % ENTRY.itemsize or size >= width * 8:
            raise ValueError('noncanonical sparse payload length')
        entries = np.frombuffer(packet, dtype=ENTRY, offset=HEADER.size)
        indices = entries['index']
        if len(indices) and (indices[-1] >= width or np.any(indices[1:] <= indices[:-1])):
            raise ValueError('sparse indices must be unique, ascending and in bounds')
        if np.any(entries['bits'] == default):
            raise ValueError('explicit default bits are noncanonical')
        # Only allocate after all attacker-controlled sizes/indices are checked.
        bits = np.full(width, default, dtype='<u8')
        bits[indices] = entries['bits']
    result = np.frombuffer(bits.tobytes(), dtype='<f8')
    if kind == 'filtered_logits': result = result.reshape(1, width)
    return DecodedVector(kind, result)


ENCODING = 'keyprint.binary64-vector.v1'


def commitment(values, kind):
    return {'encoding': ENCODING, 'kind': kind, 'width': values.size,
            'sha256': vector_digest(values, kind)}
