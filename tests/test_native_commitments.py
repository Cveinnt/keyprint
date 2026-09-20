"""New commitments cannot be mistaken for legacy hashes or acceptance."""
import copy
import random

import pytest

pytest.importorskip('keyprint_native')
from keyprint import Keyprint
from keyprint.experimental.native_mlx import NativePublicCandidate
from keyprint.experimental.native_reporting import sampling
from keyprint.experimental.fast_reporting import build_report
from keyprint._engine.research.keyprint_v3_reporting_contract import _sampling, _digest
from test_fast_mlx import head
from test_native_mlx import native_vectors
from commitment_parity import legacy_sampling_projection, validate_observed_sequence


@pytest.fixture
def response():
    key = bytes(range(32))
    facade = NativePublicCandidate(Keyprint(key=key)._candidate)
    pipe = facade._core.pipeline(key, condition='marked')
    try:
        pipe.step(head([32, 33, 34]), random.Random(12).getrandbits)
        pipe.step(head([151645]), random.Random(12).getrandbits)
        receipt = pipe.receipt()
        yield facade, pipe, facade._generation(receipt)
    finally:
        pipe.close()


def test_versioned_report_resolves_to_legacy_bytes(response):
    facade, pipe, report = response
    payload = report['payload']
    assert report['schema'] == 'keyprint.experimental-native-report.v2'
    assert payload['vector_commitment_encoding'] == 'keyprint.binary64-vector.v1'
    assert payload['vector_commitments_resolved'] is False
    assert report['verdict'] is None and report['all_reporting_surfaces_accepted'] is False
    records = payload['sampling_records']
    observed = pipe._raw._observed_vector_packets
    validate_observed_sequence(observed, 2)
    assert sampling(records, payload['committed_token_ids']) == records
    with pytest.raises(ValueError, match='fields'):
        _sampling(records, payload['committed_token_ids'])
    legacy = legacy_sampling_projection(records, observed)
    assert _sampling(legacy, payload['committed_token_ids']) == legacy
    with pytest.raises(ValueError, match='fields'):
        sampling(legacy, payload['committed_token_ids'])


@pytest.mark.parametrize('field,value', [
    ('encoding', 'raw-float64'), ('kind', 'filtered_logits'), ('width', 151668),
    ('width', True), ('width', 151669.0), ('sha256', 'z' * 64), ('sha256', '0' * 63),
    ('extra', 'unexpected'),
])
def test_malformed_commitment_rejected(response, field, value):
    _, _, report = response
    payload = report['payload']
    records = copy.deepcopy(payload['sampling_records'])
    records[0]['base_probability_commitment'][field] = value
    with pytest.raises((ValueError, TypeError)):
        sampling(records, payload['committed_token_ids'])


@pytest.mark.parametrize('mutation', ['step', 'token', 'channel', 'point', 'acceptance', 'missing', 'extra'])
def test_sampling_invariants_retained(response, mutation):
    _, _, report = response
    payload = report['payload']
    records = copy.deepcopy(payload['sampling_records'])
    if mutation == 'step': records[0]['step'] = 1
    elif mutation == 'token': records[0]['token_id'] = 0
    elif mutation == 'channel': records[0]['channel'] = 'unknown'
    elif mutation == 'point': records[0]['randomness']['integer_point_decimal'] = records[0]['randomness']['total_weight_decimal']
    elif mutation == 'acceptance': records[0]['randomness']['transcript'][-1]['accepted'] = False
    elif mutation == 'missing': records.pop()
    elif mutation == 'extra': records.append(records[-1])
    with pytest.raises((ValueError, TypeError)):
        sampling(records, payload['committed_token_ids'])


@pytest.mark.parametrize('name', ['vector_commitment.py', 'native_reporting.py'])
def test_codec_and_validator_sources_bound(response, name):
    facade, _, _ = response
    target = facade.core_identity
    target['specification']['execution']['vector_commitments']['sources'][name] = '0' * 64
    target['runtime_profile_sha256'] = _digest(target['specification'])
    with pytest.raises(ValueError, match='execution source mismatch'):
        build_report('error', target_identity=target, payload={})


@pytest.mark.parametrize('mutation', ['changed_packet', 'missing', 'reordered', 'changed_hash'])
def test_parity_needs_resolved_unchanged_bytes(response, mutation):
    _, pipe, report = response
    observed = copy.deepcopy(pipe._raw._observed_vector_packets)
    if mutation == 'changed_packet': observed[1]['packet_hex'] = observed[1]['packet_hex'][:-2] + '01'
    elif mutation == 'missing': observed.pop()
    elif mutation == 'reordered': observed[0], observed[1] = observed[1], observed[0]
    elif mutation == 'changed_hash': observed[1]['commitment']['sha256'] = '0' * 64
    with pytest.raises((ValueError, TypeError)):
        validate_observed_sequence(observed, 2)
        legacy_sampling_projection(report['payload']['sampling_records'], observed)
