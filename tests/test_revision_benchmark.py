"""Timing comparisons require complete matching paths, not just similar text."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import random

import pytest

TOOLS = Path(__file__).parents[1] / 'tools'
sys.path.insert(0, str(TOOLS))
try:
    from benchmark_revision import analyze
    from audit_revision import check_journal, report_target
finally:
    sys.path.pop(0)


def fixture():
    return [{'case': 'one', 'repeat': repeat, 'condition': condition, 'arm': arm,
             'seed': repeat, 'prompt_ids': [1], 'token_ids': [2, 3], 'text': 'ok',
             'completion': 'eos', 'completion_tokens': 2,
             'sampling_records': [{'randomness': [1, 0]}], 'reservations': {'sample': 2},
             'seconds': 1. if arm == 'baseline' else .8}
            for repeat in range(4) for condition in ('ordinary', 'marked')
            for arm in ('baseline', 'candidate')]


def test_summary_uses_paired_revision_ratio_for_each_condition():
    result = analyze(fixture(), ['one'])
    assert result['matched_paths'] and not result['production_overhead_accepted']
    for value in result['candidate_over_baseline'].values():
        assert value['pairs'] == 4
        assert value['seconds_per_committed_token']['geometric_mean_ratio'] == pytest.approx(.8)
        assert value['request_latency']['geometric_mean_ratio'] == pytest.approx(.8)
        assert not any('pass' in k for k in value)


@pytest.mark.parametrize('field,value', [('seed', 999), ('prompt_ids', [9]),
    ('token_ids', [4, 5]), ('text', 'different'), ('completion', 'length'),
    ('completion_tokens', 8), ('sampling_records', [{'randomness': [1, 1]}]),
    ('reservations', {'sample': 3})])
def test_mismatched_path_cannot_receive_a_timing_summary(field, value):
    rows = fixture()
    rows[1][field] = value
    with pytest.raises(ValueError, match='Matched-path'):
        analyze(rows, ['one'])


@pytest.mark.parametrize('bad', ['missing', 'duplicate', 'failed', 'zero_time', 'nan_time', 'wrong_count'])
def test_missing_failed_or_invalid_measurement_cannot_pass(bad):
    rows = fixture()
    if bad == 'missing': rows.pop()
    elif bad == 'duplicate': rows.append(copy.deepcopy(rows[0]))
    elif bad == 'failed': rows[0]['error'] = {'type': 'Injected'}
    elif bad == 'zero_time': rows[0]['seconds'] = 0.
    elif bad == 'nan_time': rows[0]['seconds'] = float('nan')
    elif bad == 'wrong_count': rows[0]['completion_tokens'] = rows[1]['completion_tokens'] = 1
    with pytest.raises(ValueError):
        analyze(rows, ['one'])


@pytest.mark.parametrize('corruption', [None, 'draw', 'chain', 'identity'])
def test_audit_replays_draws_and_checks_chain_before_normalizing_identities(tmp_path, corruption):
    rng = random.Random(42)
    previous, lines = '0' * 64, []
    for i, count in enumerate((8, 17, 1)):
        event = {'kind': 'random_bits_returned', 'bits': count,
                 'value_decimal': str(rng.getrandbits(count)), 'caller_sha256': 'a' * 64,
                 'runtime_profile_sha256': 'b' * 64}
        if corruption == 'draw' and i == 0:
            event['value_decimal'] = '999'
        if corruption == 'chain' and i == 1:
            previous = 'c' * 64
        if corruption == 'identity' and i == 1:
            event['caller_sha256'] = 'd' * 64
        line = (json.dumps({'sequence': i, 'previous_sha256': previous, 'event': event}) + '\n').encode()
        lines.append(line)
        previous = hashlib.sha256(line).hexdigest()
    path = tmp_path / 'journal.jsonl'
    path.write_bytes(b''.join(lines))
    if corruption:
        with pytest.raises(ValueError, match='Journal'):
            check_journal(path, 42, {'caller_sha256': 'a' * 64, 'runtime_profile_sha256': 'b' * 64})
    else:
        events = check_journal(path, 42)
        assert len(events) == 3
        assert all('caller_sha256' not in e and 'runtime_profile_sha256' not in e for e in events)


def test_compact_report_target_requires_verified_full_worker_specification():
    spec = {'version': 'fixture', 'execution': {'source': 'abc'}}
    identity = {'version': 'fixture', 'max_steps': 2048, 'score_namespace_sha256': '0' * 64,
                'deployment_calibrated': False, 'specification': spec,
                'runtime_profile_sha256': hashlib.sha256(json.dumps(
                    spec, sort_keys=True, separators=(',', ':')).encode()).hexdigest()}
    compact = report_target(identity)
    assert 'specification' not in compact and compact['specification_digest_verified'] is True
    assert compact['caller_identity_authentication'] == 'not_performed'
    assert compact['runtime_profile_sha256'] == identity['runtime_profile_sha256']
    identity['specification']['execution']['source'] = 'changed'
    with pytest.raises(ValueError, match='digest'):
        report_target(identity)
