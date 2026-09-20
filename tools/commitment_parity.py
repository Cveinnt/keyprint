"""Explicit test-only observation of native v2 vectors and legacy projection.

This instruments the native host only inside the context manager. It never
modifies reference execution. Captured packets resolve new commitments before
comparison with legacy dense hashes. Instrumented timings are not serving data.
"""
from contextlib import contextmanager
import copy
import hashlib

from vector_commitment import encode_vector, decode_vector


def observe(values, kind, result):
    # The independently retained development codec is the byte-format oracle.
    packet = encode_vector(values, kind)
    if decode_vector(packet).values.tobytes() != values.tobytes():
        raise ValueError('Observed vector did not reconstruct exactly')
    expected = {'encoding': 'keyprint.binary64-vector.v1', 'kind': kind,
                'width': values.size, 'sha256': hashlib.sha256(packet).hexdigest()}
    if result != expected:
        raise ValueError('SDK commitment differs from independently encoded bytes')
    return {'commitment': result, 'packet_hex': packet.hex()}


@contextmanager
def observe_native_vectors():
    from keyprint.experimental.native_mlx import _NativeHost
    original_prob = _NativeHost._probability_commitments
    original_filter = _NativeHost._filtered_commitment
    captured = []
    def record(host, entry):
        if not hasattr(host, '_observed_vector_packets'):
            host._observed_vector_packets = []
        host._observed_vector_packets.append(entry)
        captured.append(entry)
    def probabilities(host, base, prepared):
        result = original_prob(host, base, prepared)
        for name, values in [('base', base), ('prepared', prepared)]:
            record(host, observe(values, 'probability', result[name + '_probability_commitment']))
        return result
    def filtered(host, values):
        result = original_filter(host, values)
        record(host, observe(values, 'filtered_logits', result))
        return result
    _NativeHost._probability_commitments = probabilities
    _NativeHost._filtered_commitment = filtered
    try:
        yield captured
    finally:
        _NativeHost._probability_commitments = original_prob
        _NativeHost._filtered_commitment = original_filter


def resolve(entry, kind):
    packet = bytes.fromhex(entry['packet_hex'])
    decoded = decode_vector(packet)
    expected_width = 151669 if kind == 'probability' else 151936
    expected = {'encoding': 'keyprint.binary64-vector.v1', 'kind': kind,
                'width': expected_width, 'sha256': hashlib.sha256(packet).hexdigest()}
    if (decoded.kind != kind or decoded.values.size != expected_width
            or entry['commitment'] != expected or encode_vector(decoded.values, kind) != packet):
        raise ValueError('Vector artifact differs from its declared commitment')
    return hashlib.sha256(decoded.values.tobytes()).hexdigest()


def legacy_sampling_projection(records, observations):
    """Resolve real bytes, then project their hashes for legacy parity only."""
    probabilities = [e for e in observations if e['commitment']['kind'] == 'probability']
    if len(probabilities) != 2 * len(records):
        raise ValueError('Missing or extra observed probability vectors')
    result = copy.deepcopy(records)
    for i, record in enumerate(result):
        for offset, name in enumerate(('base', 'prepared')):
            entry = probabilities[2 * i + offset]
            if record.pop(name + '_probability_commitment') != entry['commitment']:
                raise ValueError('Sampling record differs from resolved vector')
            record[name + '_probability_sha256'] = resolve(entry, 'probability')
    return result


def validate_observed_sequence(observations, steps):
    if len(observations) != steps * 3:
        raise ValueError('Expected filter, base and prepared vectors per committed step')
    for index, entry in enumerate(observations):
        resolve(entry, 'filtered_logits' if index % 3 == 0 else 'probability')


def legacy_receipt_projection(receipt, observations):
    result = copy.deepcopy(receipt)
    result['sampling_records'] = legacy_sampling_projection(result['sampling_records'], observations)
    filtered = iter(e for e in observations if e['commitment']['kind'] == 'filtered_logits')
    for attempt in result['shared_filter_attempts']:
        if 'filtered_logits_commitment' in attempt:
            entry = next(filtered)
            if attempt.pop('filtered_logits_commitment') != entry['commitment']:
                raise ValueError('Filter attempt differs from resolved vector')
            attempt['filtered_logits_sha256'] = resolve(entry, 'filtered_logits')
    if next(filtered, None) is not None:
        raise ValueError('Extra observed filter vectors')
    return result
