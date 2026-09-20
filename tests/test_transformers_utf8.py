"""Controlled-model boundary tests; downloaded inference is qualified separately."""
import json
from types import SimpleNamespace

import pytest

torch = pytest.importorskip('torch')

from keyprint import Keyprint, KeyprintError
from keyprint.backends.bytelevel import ByteLevelBinding
from keyprint.backends.transformers import TransformersModel


class ByteTokenizer:
    def __init__(self, pieces):
        self.pieces = pieces
        self.decode_calls = 0

    def apply_chat_template(self, *args, **kwargs):
        return [1]

    def decode(self, ids, **kwargs):
        self.decode_calls += 1
        # Typical tokenizer replacement would hide an unfinished character.
        return b''.join(self.pieces[i] or b'' for i in ids).decode('utf-8', 'replace')


class SequenceModel:
    config = SimpleNamespace(max_position_embeddings=100)

    def __init__(self, ids, size):
        self.ids, self.size, self.calls = ids, size, 0

    def __call__(self, *, input_ids, past_key_values, use_cache):
        raw = torch.full((1, input_ids.shape[1], self.size), -torch.inf, dtype=torch.float32)
        raw[:, :, self.ids[self.calls]] = 0
        self.calls += 1
        return SimpleNamespace(logits=raw, past_key_values='cache')


def candidate(parts, *, eos=False):
    pieces = (None, *parts)
    ids = [*range(1, len(pieces)), *([0] if eos else [])]
    binding = ByteLevelBinding(pieces, frozenset({0}), 'controlled-byte-binding')
    tokenizer = ByteTokenizer(pieces)
    model = SequenceModel(ids, len(pieces))
    kp = Keyprint(key=bytes(range(32)))
    kp._backend = TransformersModel(model, tokenizer, binding, {}, temperature=.7, top_k=len(pieces))
    return kp, model, tokenizer


@pytest.mark.parametrize('condition', ['ordinary', 'marked'])
@pytest.mark.parametrize('parts,text,pending', [
    ([b'A', b'\xc3'], 'A', 'c3'),
    ([b'\xc3'], '', 'c3'),
    ([b'A', b'\xe2\x9c'], 'A', 'e29c'),
    ([b'\xf0', b'\x9f', b'\x8c'], '', 'f09f8c'),
    ([b'A', b'\xe2', b'\x9c', b'\x93'], 'A✓', ''),
    ([b'\xf0\x9f', b'\x8c\xb1'], '🌱', ''),
    ([b'A', b'B'], 'AB', ''),
])
def test_exact_cap_retains_all_tokens_and_bytes(condition, parts, text, pending, tmp_path):
    kp, model, tokenizer = candidate(parts)
    result = kp.generate('prompt', condition=condition, max_tokens=len(parts), output=tmp_path / 'run')
    report = result.report
    assert result.text == text
    assert report['completion'] == 'length'
    assert model.calls == len(parts) == report['usage']['completion_tokens']
    assert report['committed_token_ids'] == list(range(1, len(parts) + 1))
    carrier = report['carrier_rendering'][0]
    assert carrier['committed_token_ids'] == report['committed_token_ids']
    assert carrier['pending_utf8_hex'] == pending
    assert text.encode() + bytes.fromhex(pending) == b''.join(parts)
    assert carrier['status'] == ('incomplete_utf8_at_token_limit' if pending else 'complete_utf8')
    assert tokenizer.decode_calls == (0 if pending else 1)
    if pending:
        assert report['literal_replay_status'][0]['availability'] == 'unavailable'
        assert report['tokenizer_rendering_check'] == 'unavailable_for_incomplete_utf8_carrier'
    assert json.loads((result.artifacts / 'report.json').read_text()) == report
    events = [json.loads(line)['event'] for line in (result.artifacts / 'journal.jsonl').read_text().splitlines()]
    assert [e['token_id'] for e in events if e['phase'] == 'committed'] == report['committed_token_ids']
    assert events[-1]['phase'] == 'complete'


@pytest.mark.parametrize('parts,eos', [([b'\xc3'], True), ([b'\xff'], False),
    ([b'\xed\xa0\x80'], False), ([b'\xe2', b'A'], False), ([b'\xf4\x90'], False)])
def test_eos_partial_and_invalid_utf8_still_fail_with_consumed_work(parts, eos, tmp_path):
    kp, model, tokenizer = candidate(parts, eos=eos)
    with pytest.raises(KeyprintError) as caught:
        kp.generate('prompt', max_tokens=len(parts) + int(eos), output=tmp_path / 'run')
    report = caught.value.report
    assert report['failed_phase'] == 'render'
    assert report['error_type'] == 'UnicodeDecodeError'
    assert report['usage']['completion_tokens'] == len(parts) + int(eos) == model.calls
    assert len(report['committed_token_ids']) == model.calls


@pytest.mark.parametrize('ids,cap', [([1], 2), ([1], True), ([], 0), ([0], 1), ([-1], 1), ([2], 1)])
def test_length_finalization_rejects_wrong_cap_eos_and_unbound_ids(ids, cap):
    binding = ByteLevelBinding((None, b'A'), frozenset({0}), 'fixture')
    with pytest.raises(ValueError, match='exact token cap'):
        binding.render_at_limit(ids, cap)


def test_complete_rendering_still_checks_tokenizer_agreement(tmp_path):
    kp, model, tokenizer = candidate([b'A'])
    tokenizer.decode = lambda *args, **kwargs: 'changed'
    with pytest.raises(KeyprintError) as caught:
        kp.generate('prompt', max_tokens=1, output=tmp_path / 'run')
    assert caught.value.report['error_type'] == 'ValueError'
    assert model.calls == 1
