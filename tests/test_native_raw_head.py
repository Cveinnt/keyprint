"""Shared raw-head commitments preserve immutable bytes and filter failures."""
from dataclasses import FrozenInstanceError
import hashlib
import json

import numpy as np
import pytest

pytest.importorskip('keyprint_native')
from keyprint import Keyprint
from keyprint.experimental.native_mlx import _RawHead, NativePublicCandidate
from test_fast_caller import Backend, head


@pytest.mark.parametrize('strided', [False, True])
def test_snapshot_keeps_exact_bits_without_mutable_aliases(strided):
    bits = np.array([0, 0x80000000, 1, 0x7fc00042, 0xff800000, 0x7f800000], dtype=np.uint32)
    values = bits.view(np.float32).reshape(1, -1)
    if strided:
        values = values[:, ::-1]
    expected = values.tobytes()
    snapshot = _RawHead(values)
    values[:] = 7
    assert snapshot.values.tobytes() == expected
    assert snapshot.sha256 == hashlib.sha256(expected).hexdigest()
    with pytest.raises(ValueError):
        snapshot.values.setflags(write=True)
    with pytest.raises(FrozenInstanceError):
        snapshot.sha256 = '0' * 64
    with pytest.raises(TypeError):
        _RawHead(values, sha256='0' * 64)
    view = snapshot.values
    with pytest.warns(DeprecationWarning, match='Setting the shape'):
        view.shape = (view.size,)
    assert snapshot.values.shape == (1, len(bits))


@pytest.mark.parametrize('values', [None, [1.], np.zeros((1, 4), dtype=np.float64),
    np.zeros(4, dtype=np.float32), np.zeros((2, 4), dtype=np.float32),
    np.zeros((1, 0), dtype=np.float32), np.zeros((1, 262145), dtype=np.float32)])
def test_snapshot_rejects_wrong_type_shape_or_bound(values):
    with pytest.raises(ValueError, match='single-row float32'):
        _RawHead(values)


@pytest.mark.parametrize('condition', ['ordinary', 'marked'])
def test_complete_caller_hashes_raw_bytes_once_and_retains_both_receipts(tmp_path, monkeypatch, condition):
    kp = Keyprint(key=bytes(range(32)))
    kp._candidate = NativePublicCandidate(kp._candidate)
    raw = head(np.array([[32]]))
    expected_bytes = raw.tobytes()
    real_sha = hashlib.sha256
    expected_hash = real_sha(expected_bytes).hexdigest()
    hashes = []
    receipts = capture_receipts(kp, monkeypatch)

    def counting_sha(data=b'', *args, **kwargs):
        if data == expected_bytes:
            hashes.append(1)
        return real_sha(data, *args, **kwargs)

    monkeypatch.setattr(hashlib, 'sha256', counting_sha)
    result = kp._run(lambda *_args, **_kwargs: raw.copy(), [32], max_tokens=1,
                     condition=condition, output=tmp_path / 'run',
                     backend=Backend, cache_factory=lambda _: [])
    assert len(hashes) == 1
    events = [json.loads(line)['event'] for line in (result.artifacts / 'journal.jsonl').read_text().splitlines()]
    prepared, = [e for e in events if e['kind'] == 'prepared_step']
    assert prepared['raw_logits_sha256'] == expected_hash
    attempt, = receipts[-1]['shared_filter_attempts']
    assert attempt['raw_logits_sha256'] == expected_hash


@pytest.mark.parametrize('bad_token', [32, 151935])
@pytest.mark.parametrize('bad_value', [np.nan, np.inf])
def test_snapshot_does_not_skip_filter_validation_or_consume_randomness(bad_token, bad_value):
    kp = Keyprint(key=bytes(range(32)))
    candidate = NativePublicCandidate(kp._candidate)._core
    raw = head(np.array([[32]]))[0]
    raw[0, bad_token] = bad_value
    draws = []
    pipeline = candidate.pipeline(bytes(range(32)), condition='marked')
    try:
        with pytest.raises(ValueError, match='NaN and positive infinity'):
            pipeline.step(_RawHead(raw), lambda k: draws.append(k) or 0)
        assert draws == [] and pipeline.committed_token_ids == ()
        attempt, = pipeline.receipt()['shared_filter_attempts']
        assert attempt['status'] == 'failed_before_commit'
        assert attempt['raw_logits_sha256'] == hashlib.sha256(raw.tobytes()).hexdigest()
    finally:
        pipeline.close()


def test_grammar_mask_uses_its_own_snapshot_and_hash(tmp_path, monkeypatch):
    from test_mlx_structured import candidate, SCHEMA
    kp, ids, calls = candidate(monkeypatch, execution='experimental-native')
    receipts = capture_receipts(kp, monkeypatch)
    result = kp.generate('JSON', json_schema=SCHEMA, max_tokens=64, output=tmp_path / 'json')
    assert calls == ids and result.report['structured_output']['schema_validated']
    events = [json.loads(line)['event'] for line in (result.artifacts / 'journal.jsonl').read_text().splitlines()]
    raw = [e['raw_logits_sha256'] for e in events if e['kind'] == 'prepared_step']
    masked = [e['masked_logits_sha256'] for e in events if e['kind'] == 'grammar_mask']
    filters = [e['raw_logits_sha256'] for e in receipts[-1]['shared_filter_attempts']]
    assert len(raw) == len(masked) == len(filters) == len(ids)
    assert all(a != b for a, b in zip(raw, masked, strict=True))
    assert masked == filters


def capture_receipts(kp, monkeypatch):
    receipts = []
    original = kp._candidate._generation

    def retain(receipt, **kwargs):
        receipts.append(receipt)
        return original(receipt, **kwargs)

    monkeypatch.setattr(kp._candidate, '_generation', retain)
    return receipts
