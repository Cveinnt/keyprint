"""Corrupted observations must not count as matching exported probabilities."""
import copy
import hashlib
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / 'tools'))
try:
    from audit_vector_commitments import check_vectors
    from vector_commitment import encode_vector
finally:
    sys.path.pop(0)


@pytest.fixture
def observations():
    logits = np.full((1, 151936), -np.inf)
    logits[0, 17] = 0.
    base = np.zeros(151669)
    base[17:19] = [.25, .75]
    prepared = np.zeros(151669)
    prepared[17:19] = [.5, .5]
    entries = []
    for kind, values in [('filtered_logits', logits), ('probability', base), ('probability', prepared)]:
        packet = encode_vector(values, kind)
        entries.append({'kind': kind, 'packet_hex': packet.hex(), 'dense_bytes': values.nbytes,
                        'encoded_bytes': len(packet), 'dense_sha256': hashlib.sha256(values.tobytes()).hexdigest(),
                        'encoded_sha256': hashlib.sha256(packet).hexdigest()})
    samples = [{'base_probability_sha256': entries[1]['dense_sha256'],
                'prepared_probability_sha256': entries[2]['dense_sha256']}]
    return entries, samples


def test_complete_observations(observations):
    entries, samples = observations
    result = check_vectors(entries, samples)
    assert result['vectors'] == 3
    assert result['dense_bytes'] == sum(e['dense_bytes'] for e in entries)


@pytest.mark.parametrize('mutation', ['missing', 'extra', 'reordered', 'packet', 'dense_hash',
                                     'encoded_hash', 'size', 'kind', 'report', 'empty'])
def test_corruption_rejected(observations, mutation):
    entries, samples = copy.deepcopy(observations)
    if mutation == 'missing': entries.pop()
    elif mutation == 'extra': entries.append(entries[-1])
    elif mutation == 'reordered': entries[1], entries[2] = entries[2], entries[1]
    elif mutation == 'packet': entries[1]['packet_hex'] = entries[1]['packet_hex'][:-2] + '01'
    elif mutation == 'dense_hash': entries[1]['dense_sha256'] = '0' * 64
    elif mutation == 'encoded_hash': entries[1]['encoded_sha256'] = '0' * 64
    elif mutation == 'size': entries[1]['encoded_bytes'] += 1
    elif mutation == 'kind': entries[1]['kind'] = 'filtered_logits'
    elif mutation == 'report': samples[0]['base_probability_sha256'] = '0' * 64
    elif mutation == 'empty': entries, samples = [], []
    with pytest.raises(ValueError):
        check_vectors(entries, samples)
