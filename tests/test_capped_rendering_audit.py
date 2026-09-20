import copy
import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from audit_capped_rendering import verify_rendering


def fixture(completion='length', pending=True):
    token_bytes = [b'A', b'\xe2\x9c', None]
    tokens = [0, 2 if completion == 'eos' else 1 if pending else 0]
    visible = tokens[:-1] if completion == 'eos' else tokens
    partial = completion == 'length' and pending
    text = 'A' if partial or completion == 'eos' else 'AA'
    records = [{'step': i, 'channel': 'visible', 'token_id': token,
                'randomness': {'total_weight_decimal': '3', 'integer_point_decimal': '0',
                               'transcript': [{'bit_count': 2, 'value_decimal': '0', 'accepted': True}]}}
               for i, token in enumerate(tokens)]
    report = {'payload': {'completion': completion, 'committed_token_ids': tokens,
        'sampling_records': records,
        'literal_diagnostics': [{'identity': {'text_sha256': hashlib.sha256(text.encode()).hexdigest()},
            'availability': 'statistic_unavailable', 'events': None, 'ones': None, 'trials': None,
            'random_key_reference_tail': {'value': None}}]},
        'rendered_carriers': {'visible_text': text, 'reasoning_text': None, 'tool_texts': [], 'protocol_complete': True},
        'carrier_rendering': [{'channel': 'visible', 'committed_token_ids': visible,
            'status': 'incomplete_utf8_at_token_limit' if partial else 'complete_utf8',
            'pending_utf8_hex': 'e29c' if partial else ''}],
        'literal_replay_status': [{'channel': 'visible', 'availability': 'unavailable', 'reason': 'fixture'}]}
    events = [{'kind': 'response_started', 'purpose': 'general', 'max_tokens': 2},
              {'kind': 'model_forward_reserved', 'prefill': True}]
    for i in range(2):
        events.append({'kind': 'model_forward_reserved', 'prefill': False})
        for kind in ('random_bits_requested', 'random_bits_returned'):
            events.append({'kind': kind, 'index': i, 'sample_index': i, 'bits': 2, 'value_decimal': '0'})
        events.append({'kind': 'committed_step', 'index': i})
    events.append({'kind': 'response_terminal', 'outcome': completion, 'bit_requests': 2,
                   'bit_values_obtained': 2, 'bit_values_journaled': 2, 'bits_requested': 4,
                   'model_calls': 3, 'prefill_calls': 1, 'sample_attempts': 2})
    return report, events, token_bytes, 2


@pytest.mark.parametrize('completion,pending', [('length', True), ('length', False), ('eos', False)])
def test_reconciles_visible_bytes_and_exact_draws(completion, pending):
    result = verify_rendering(*fixture(completion, pending))
    assert result == {'pending_utf8_bytes': 2 if pending else 0, 'verified_random_draws': 2}


@pytest.mark.parametrize('mutation', [
    'text', 'lost_suffix', 'replacement', 'ids', 'control', 'available_score',
    'draw', 'draw_order', 'counter', 'cap', 'completion', 'rejection',
])
def test_rejects_lost_bytes_or_inconsistent_work(mutation):
    report, events, token_bytes, eos = copy.deepcopy(fixture())
    if mutation == 'text': report['rendered_carriers']['visible_text'] = 'different'
    elif mutation == 'lost_suffix': report['carrier_rendering'][0]['pending_utf8_hex'] = ''
    elif mutation == 'replacement': report['rendered_carriers']['visible_text'] += '\ufffd'
    elif mutation == 'ids': report['carrier_rendering'][0]['committed_token_ids'] = [0]
    elif mutation == 'control': token_bytes[1] = None
    elif mutation == 'available_score': report['payload']['literal_diagnostics'][0]['events'] = 1
    elif mutation == 'draw': events[4]['value_decimal'] = '1'
    elif mutation == 'draw_order': events[3], events[4] = events[4], events[3]
    elif mutation == 'counter': events[-1]['bit_values_journaled'] = 1
    elif mutation == 'cap': events[0]['max_tokens'] = 3
    elif mutation == 'completion': report['payload']['completion'] = 'eos'
    elif mutation == 'rejection': report['payload']['sampling_records'][0]['randomness']['transcript'][0]['accepted'] = False
    with pytest.raises(ValueError):
        verify_rendering(report, events, token_bytes, eos)


@pytest.mark.parametrize('condition', ['ordinary', 'marked'])
@pytest.mark.parametrize('token', [32, 25521, (32, 33)])
def test_audits_actual_sdk_report_serialization(tmp_path, condition, token):
    from test_fast_caller import candidate, Backend, head
    from analyze_mlx_capacity import read_journal
    from keyprint._engine.legacy._impl.research.token_channel_host import EOS
    sdk = candidate()
    generated = sdk._run(lambda ids, **_: head(ids, token if isinstance(token, tuple) else (token,)), [32], max_tokens=1,
        condition=condition, output=tmp_path / 'response', backend=Backend, cache_factory=lambda _: [])
    report = json.loads((generated.artifacts / 'report.json').read_text())['report']
    events = read_journal(generated.artifacts / 'journal.jsonl')
    token_bytes = sdk._candidate._core.reference._base._binding.token_bytes
    result = verify_rendering(report, events, token_bytes, EOS)
    assert result['pending_utf8_bytes'] == (2 if token == 25521 else 0)
    if isinstance(token, tuple):
        assert result['verified_random_draws'] >= 1
    else:
        assert result['verified_random_draws'] == 0
