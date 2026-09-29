import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / 'tools'))
from balanced_inference import generate, audit_output
from balanced_source_session import BalancedProfile, Config

KEY = bytes(range(32))


def fixture(tmp_path, **changes):
    binding = SimpleNamespace(pieces=(b'a', b' b', None), eos_ids=frozenset({2}))
    profile = BalancedProfile(binding.pieces, tokenizer_identity='fake-native-fixture',
                           eos_ids=[2], config=Config(max_steps=10))
    calls = []
    def forward(ids):
        calls.append(list(ids))
        return np.array([[2., 1., -20.]] if len(calls) == 1 else [[-np.inf, -np.inf, 0.]], dtype=np.float32)
    args = dict(profile=profile, binding=binding, key=KEY, condition='marked',
                prompt_ids=[0, 1], forward=forward, decode=lambda ids: ''.join('a' if i == 0 else ' b' for i in ids),
                random_bits=lambda count: 0, output=tmp_path/'run', max_tokens=10)
    args.update(changes)
    return args, calls


@pytest.mark.parametrize('condition', ['ordinary', 'marked'])
def test_native_callback_receives_prompt_then_exact_committed_tokens(tmp_path, condition):
    args, calls = fixture(tmp_path, condition=condition)
    result = generate(**args)
    assert calls == [[0, 1], [0]]
    assert result['completion'] == 'eos' and result['committed_token_ids'] == [0, 2]
    assert result['text'] == 'a' and result['model_calls'] == 2
    verified = audit_output(args['output'], args['profile'], KEY, binding=args['binding'])
    assert verified['verified_draws'] == 2 and not verified['native_heads_replayed']


def test_cap_is_retained_not_silently_retried_or_called_eos(tmp_path):
    args, calls = fixture(tmp_path, max_tokens=1)
    result = generate(**args)
    assert len(calls) == 1 and result['completion'] == 'cap'
    assert audit_output(args['output'], args['profile'], KEY, binding=args['binding'])['verified_draws'] == 1


def test_forward_failure_retains_partial_text_and_commits(tmp_path):
    args, calls = fixture(tmp_path); original = args['forward']
    def fail(ids):
        if calls: raise RuntimeError('fixture native forward failed')
        return original(ids)
    args['forward'] = fail; result = generate(**args)
    assert result['completion'] == 'incomplete' and result['text'] == 'a'
    assert result['failed_phase'] == 'forward' and result['model_calls'] == 2
    assert result['committed_token_ids'] == [0]
    with pytest.raises(ValueError): audit_output(args['output'], args['profile'], KEY, binding=args['binding'])


@pytest.mark.parametrize('raw', [np.array([[np.nan, 0., 0.]], np.float32),
                                 np.array([[np.inf, 0., 0.]], np.float32),
                                 np.array([[0., 0., 0.]], np.float64)])
def test_invalid_native_heads_leave_failure_receipt(tmp_path, raw):
    args, _ = fixture(tmp_path, forward=lambda ids: raw)
    result = generate(**args)
    assert result['failed_phase'] == 'filter' and result['committed_token_ids'] == []
    assert result['model_calls'] == 1


def test_entropy_failure_is_retained_without_redraw(tmp_path):
    def bits(count): raise OSError('fixture entropy failed')
    args, calls = fixture(tmp_path, random_bits=bits); result = generate(**args)
    assert result['failed_phase'] == 'draw' and len(calls) == 1
    assert result['failed_draw'] == {'transcript': [], 'callback_calls': 1}
    assert result['committed_token_ids'] == []


def test_decode_failure_cannot_hide_successful_token_trace(tmp_path):
    def decode(ids): raise UnicodeError('fixture decoder failed')
    args, _ = fixture(tmp_path, decode=decode); result = generate(**args)
    assert result['completion'] == 'eos' and result['committed_token_ids'] == [0, 2]
    assert result['decode_error']['type'] == 'UnicodeError' and 'text' not in result
    with pytest.raises(ValueError): audit_output(args['output'], args['profile'], KEY, binding=args['binding'])


def test_decoder_cannot_silently_translate_or_rewrite_sampled_text(tmp_path):
    args, _ = fixture(tmp_path, decode=lambda ids: 'rewritten text')
    result = generate(**args)
    assert result['decode_error']['message'] == 'Decoder changed committed token bytes'
    assert result['rejected_decoder_text'] == 'rewritten text' and 'text' not in result
    assert result['committed_token_ids'] == [0, 2]


def test_unicode_fragments_and_whitespace_reach_decoder_unmodified(tmp_path):
    args, _ = fixture(tmp_path)
    pieces = (b'\xc3', b'\xa9\n  ', None)
    args['binding'] = SimpleNamespace(pieces=pieces, eos_ids={2})
    args['profile'] = BalancedProfile(pieces, tokenizer_identity='utf8-fixture', eos_ids=[2], config=Config(max_steps=10))
    ids = iter([0, 1, 2])
    def forward(_):
        raw = np.full((1, 3), -np.inf, dtype=np.float32); raw[0, next(ids)] = 0.; return raw
    args.update(forward=forward, decode=lambda out: b''.join(pieces[i] for i in out).decode())
    result = generate(**args)
    assert result['text'] == 'é\n  ' and result['committed_token_ids'] == [0, 1, 2]
    assert audit_output(args['output'], args['profile'], KEY, binding=args['binding'])['verified_draws'] == 3


def test_existing_attempt_is_never_overwritten(tmp_path):
    args, _ = fixture(tmp_path); generate(**args)
    before = (args['output']/'result.json').read_bytes()
    with pytest.raises(FileExistsError): generate(**args)
    assert (args['output']/'result.json').read_bytes() == before


def test_binding_change_rejected_before_forward(tmp_path):
    args, calls = fixture(tmp_path)
    args['binding'] = SimpleNamespace(pieces=(b'b', b'a', None), eos_ids={2})
    with pytest.raises(ValueError, match='Binding'): generate(**args)
    assert not calls and not args['output'].exists()


@pytest.mark.parametrize('field,value', [('max_tokens', 11), ('max_tokens', True),
                                        ('top_k', 0), ('prompt_ids', [-1]), ('temperature', 0.)])
def test_invalid_settings_rejected_without_model_call(tmp_path, field, value):
    args, calls = fixture(tmp_path, **{field: value})
    with pytest.raises(ValueError): generate(**args)
    assert calls == []


@pytest.mark.parametrize('tamper', ['weights', 'draw', 'order', 'ending', 'tokens', 'text'])
def test_replay_rejects_tampering_even_after_journal_hash_refresh(tmp_path, tamper):
    args, _ = fixture(tmp_path); result = generate(**args)
    journal = args['output']/'journal.jsonl'
    rows = [json.loads(l) for l in journal.read_text().splitlines()]
    if tamper == 'weights': rows[1]['distribution']['weights'][0] += 1
    if tamper == 'draw': rows[2]['draw']['token_index'] = 1
    if tamper == 'order': rows[2], rows[3] = rows[3], rows[2]
    if tamper == 'ending': result['completion'] = 'cap'
    if tamper == 'tokens': result['committed_token_ids'][0] = 1
    if tamper == 'text': result['text'] = 'altered meaning'
    journal.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    result['journal_sha256'] = hashlib.sha256(journal.read_bytes()).hexdigest()
    (args['output']/'result.json').write_text(json.dumps(result))
    with pytest.raises(ValueError): audit_output(args['output'], args['profile'], KEY, binding=args['binding'])

