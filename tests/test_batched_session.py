"""Session parity and snapshot ownership after support-mask reuse."""
import numpy as np
import pytest

from keyprint.experimental import batched_tournament as candidate
from keyprint._engine.legacy._impl.research.grouped_canonical_prototype import Config, Profile
from keyprint._engine.legacy._impl.research.token_source_sparse_execution import SparseTokenSourceSession
from keyprint._engine.legacy._impl.research.byte_trie_source_policy import SourceRequest


@pytest.fixture(params=[12, 151669])
def profile(request):
    pieces = [b'A', b'B', None, b' A', b'\t', 'é'.encode()]
    pieces += [None] * (request.param - len(pieces) - 1) + [b'C']
    return Profile(pieces, tokenizer_identity='session-snapshot-test', config=Config(layers=2))


@pytest.mark.parametrize('mode', ['ordinary', 'marked', 'repeated', 'startup', 'protected'])
def test_bytes_support_and_ownership_match_frozen_session(profile, mode):
    request = SourceRequest('proofread', 'A' * 128) if mode in ('startup', 'protected') else SourceRequest()
    condition = 'ordinary' if mode == 'ordinary' else 'marked'
    sessions = [cls(profile, bytes(range(32)), condition=condition, request=request)
                for cls in (SparseTokenSourceSession, candidate.BatchedTokenSourceSession)]
    try:
        if mode in ('repeated', 'protected'):
            q = np.zeros(len(profile.classes)); q[0] = 1.
            for _ in range(45):
                for session in sessions:
                    session.commit(session.prepare(q), 0)
        q = np.zeros(len(profile.classes)); q[:3] = [.5, .25, .25]
        q[4] = -0.; q[-1] = np.nextafter(0., 1.)
        before = q.tobytes()
        prepared = [s.prepare(q) for s in sessions]
        assert q.tobytes() == before
        assert prepared[0].probabilities.tobytes() == prepared[1].probabilities.tobytes()
        if mode in ('ordinary', 'repeated', 'startup'):
            assert prepared[1].probabilities.tobytes() == before
        if mode == 'protected':
            assert sessions[1].last_decision['mode'] == 'source_protected'
        assert sessions[0]._support.tobytes() == sessions[1]._support.tobytes()
        assert sessions[0].source_receipt() == sessions[1].source_receipt()
        # Neither caller mutation nor toggling WRITEABLE can alter prepared data.
        q[:] = 0.
        for item in prepared:
            with pytest.raises(ValueError): item.probabilities.setflags(write=True)
            assert item.probabilities[0] > 0
        for s, item in zip(sessions, prepared):
            with pytest.raises(ValueError, match='zero prepared probability'):
                s.commit(item, 4)
        assert sessions[0].commit(prepared[0], 0) == sessions[1].commit(prepared[1], 0)
    finally:
        for session in sessions: session.close()


@pytest.mark.parametrize('fault', ['nan', 'positive_inf', 'negative', 'zero_mass', 'changed_support'])
def test_transformed_vector_guard_rejects_before_pending_commit(monkeypatch, fault):
    profile = Profile([b'A', b'B', b'C'], tokenizer_identity='bad-transform-test')
    session = candidate.BatchedTokenSourceSession(profile, bytes(32), condition='marked')
    q = np.array([.5, .5, 0.])
    def corrupted(*args, **kwargs):
        out = q.copy()
        if fault == 'nan': out[0] = np.nan
        elif fault == 'positive_inf': out[0] = np.inf
        elif fault == 'negative': out[0] = -.5
        elif fault == 'zero_mass': out[:] = 0.
        else: out[:] = [.5, 0., .5]
        return out
    monkeypatch.setattr(candidate, 'transform', corrupted)
    try:
        with pytest.raises(ArithmeticError, match='invalid transformed probability vector or support'):
            session.prepare(q)
        assert session._pending is None and session._support is None and session._steps == 0
        assert session.source_receipt()['decision_counts']['full_mark'] == 0
    finally:
        session.close()
