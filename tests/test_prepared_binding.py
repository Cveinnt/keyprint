"""Prepared construction preserves state, failures, ownership and channel law."""
from concurrent.futures import ThreadPoolExecutor
import random

import numpy as np
import pytest

pytest.importorskip('keyprint_native')
from keyprint.experimental.native_mlx import NativeCandidate, _NativeHost
from keyprint.experimental.prepared_binding import PreparedBinding, boundary
from keyprint._engine.research.keyprint_candidate_v3.adapter import Candidate
from keyprint._engine.legacy._impl.research.byte_trie_source_policy import SourceRequest
from keyprint._engine.legacy._impl.research.token_channel_host import (
    ChannelRequest, THINK_OPEN, THINK_CLOSE, TOOL_OPEN, TOOL_CLOSE, EOS,
)
from test_fast_mlx import head, assert_parity
from test_native_mlx import native_vectors


@pytest.fixture(scope='module')
def candidate():
    return NativeCandidate(Candidate())


@pytest.mark.parametrize('purpose,source', [('general', None), ('proofread', 'Hello world'),
                                         ('transform', 'Bonjour')])
def test_prepared_constructor_matches_full_constructor(candidate, purpose, source):
    settings = dict(condition='marked', binding=candidate.reference._base._binding,
                    request=SourceRequest(purpose, source), channels=ChannelRequest(True, True),
                    v2_identity=candidate.identity, filter_settings=candidate.filter_settings,
                    native=candidate._native)
    full = _NativeHost(bytes(range(32)), **settings)
    prepared = _NativeHost(bytes(range(32)), prepared=candidate._prepared, **settings)
    try:
        assert full.__dict__.keys() == prepared.__dict__.keys()
        for name in full.__dict__:
            if name in ('_visible', '_current', '_excluded'):
                continue
            assert getattr(full, name) == getattr(prepared, name), name
        assert np.array_equal(full._excluded, prepared._excluded)
        assert not prepared._excluded.flags.writeable
        for name in ('ids', 'events', 'text', 'closed', 'finished', 'name'):
            assert getattr(full._visible, name) == getattr(prepared._visible, name)
        assert full._visible.decoder.getstate() == prepared._visible.decoder.getstate()
        assert full._visible.session.source_receipt() == prepared._visible.session.source_receipt()
    finally:
        full._close(); prepared._close()


@pytest.mark.parametrize('condition', ['ordinary', 'marked'])
def test_channels_and_request_state_are_isolated(candidate, condition):
    a = candidate.reference.pipeline(bytes(range(32)), condition=condition,
                                     allow_thinking=True, allow_tools=True)
    b = candidate.pipeline(bytes(range(32)), condition=condition,
                           allow_thinking=True, allow_tools=True)
    try:
        for token in (THINK_OPEN, 32, THINK_CLOSE, TOOL_OPEN, 33, TOOL_CLOSE, 34, EOS):
            for pipe in (a, b):
                pipe.step(head([token]), random.Random(1).getrandbits)
        assert_parity(a, b)
        assert a.finish().visible.text == b.finish().visible.text
        assert len({id(c.session.profile._state) for c in b._raw.carriers}) == 3
        assert all(not c.session.profile._state for c in b._raw.carriers)
    finally:
        a.close(); b.close()


@pytest.mark.parametrize('field,value', [('digest', 'bad'), ('_domain', b'bad'),
    ('tokenizer_identity', 'bad'), ('score_namespace_sha256', 'bad'),
    ('original_max_steps', 1), ('classes', (b'bad',))])
def test_mutated_profile_rejected(candidate, field, value):
    profile = candidate._prepared.profile
    old = getattr(profile, field)
    object.__setattr__(profile, field, value)
    try:
        with pytest.raises(ValueError, match='binding changed'):
            candidate.pipeline(bytes(32), condition='ordinary')
    finally:
        object.__setattr__(profile, field, old)


@pytest.mark.parametrize('field', ['history', 'layers', 'max_steps'])
def test_mutated_config_rejected(candidate, field):
    config = candidate._prepared.profile.config
    old = getattr(config, field)
    object.__setattr__(config, field, 1)
    try:
        with pytest.raises(ValueError, match='binding changed'):
            candidate.pipeline(bytes(32), condition='ordinary')
    finally:
        object.__setattr__(config, field, old)


def test_replaced_mapping_rejected(candidate, monkeypatch):
    binding = candidate.reference._base._binding
    monkeypatch.setattr(binding, '_pieces', tuple(list(binding.token_bytes)))
    with pytest.raises(ValueError, match='binding changed'):
        candidate.pipeline(bytes(32), condition='ordinary')


@pytest.mark.parametrize('name', ['CONFIG_PATH', 'TOKENIZER_PATH'])
def test_disk_corruption_after_preparation_rejected(candidate, monkeypatch, tmp_path, name):
    fake = tmp_path/'corrupt.json'
    fake.write_text('{}')
    monkeypatch.setattr(boundary, name, fake)
    with pytest.raises(ValueError, match='configuration hash mismatch'):
        candidate.pipeline(bytes(32), condition='ordinary')


def test_changed_marker_rejected(candidate, monkeypatch):
    class BrokenTokenizer:
        def id_to_token(self, token): return 'wrong'
    monkeypatch.setattr(candidate.reference._base._binding, '_tokenizer', BrokenTokenizer())
    with pytest.raises(ValueError, match='channel token mapping'):
        candidate.pipeline(bytes(32), condition='ordinary')


def test_mutable_mapping_cannot_be_prepared(candidate, monkeypatch):
    binding = candidate.reference._base._binding
    monkeypatch.setattr(binding, '_pieces', list(binding.token_bytes))
    with pytest.raises(ValueError, match='immutable tuple'):
        PreparedBinding(binding)


def test_parallel_requests_have_fresh_keys_owners_and_sessions(candidate):
    def run(n):
        key = bytes([n])*32
        pipe = candidate.pipeline(key, condition='marked')
        try:
            for _ in range(3): pipe.step(head([32, 33, 34]), random.Random(n).getrandbits)
            assert pipe._raw._key == key
            pipe.finish()
            return pipe
        finally:
            pipe.close()
    with ThreadPoolExecutor(max_workers=4) as pool:
        pipes = list(pool.map(run, range(8)))
    for name in ('_all_ids', '_control_ids', '_visible', 'sampling_records', 'filter_attempts'):
        assert len({id(getattr(p._raw, name)) for p in pipes}) == 8
    assert all(not p._raw._visible.session.profile._state for p in pipes)
    with pytest.raises(RuntimeError, match='another thread'):
        pipes[0].step(head([32]), random.Random(1).getrandbits)
