import importlib.util
from pathlib import Path
import random

import numpy as np
import pytest

from keyprint._engine.legacy._impl.research.grouped_canonical_prototype import Config, Profile
from keyprint._engine.legacy._impl.research.token_source_sparse_execution import SparseTokenSourceSession
from keyprint._engine.research.keyprint_candidate_v3.adapter import Candidate
from keyprint._engine.legacy._impl.research.token_channel_host import EOS, THINK_OPEN
from keyprint.backends.mlx_bounded import BoundedReferenceCandidate

spec = importlib.util.spec_from_file_location('mixture_candidate', Path(__file__).parents[1] / 'tools/mixture_candidate.py')
mixture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mixture)
runner_spec = importlib.util.spec_from_file_location('mixture_research_runner',
    Path(__file__).parents[1] / 'tools/mixture_research_runner.py')
runner = importlib.util.module_from_spec(runner_spec)
runner_spec.loader.exec_module(runner)


def test_mix_preserves_subnormal_support_and_excluded_mass():
    tiny = np.nextafter(0., 1.)
    base = np.array([.2, .3, .5, tiny, 0.])
    marked = np.array([.2, .7, .1, tiny, 0.])
    actual = mixture.half_mixture(base, marked)
    assert actual[0] == base[0]
    assert actual[3] == tiny and actual[4] == 0
    assert np.allclose(actual[:3], [.2, .5, .3])
    assert .5 * np.abs(actual - base).sum() == pytest.approx(.5 * (.5 * np.abs(marked - base).sum()))


@pytest.mark.parametrize('condition', ['ordinary', 'marked'])
def test_actual_session_is_half_reference_and_keeps_repeated_context_identity(condition):
    profile = Profile([None, b'a', b'b', b'c'], tokenizer_identity='mixture-fixture',
                      config=Config(history=1, max_steps=8))
    sessions = [cls(profile, bytes(range(32)), condition=condition)
                for cls in (SparseTokenSourceSession, mixture.MixtureSession)]
    base = np.array([.1, .2, .3, .4])
    try:
        for i in range(5):
            ref, mixed = [s.prepare(base) for s in sessions]
            expected = base if condition == 'ordinary' else (base + ref.probabilities) * .5
            assert mixed.probabilities.tobytes() == expected.tobytes()
            assert not mixed.probabilities.flags.writeable
            assert mixed.probabilities[0] == base[0]
            for session, prepared in zip(sessions, (ref, mixed)):
                session.commit(prepared, 1)
            if i >= 2:
                assert mixed.probabilities.tobytes() == base.tobytes()
    finally:
        for s in sessions:
            s.close()


def head(tokens):
    raw = np.full((1, 151936), -np.inf, dtype=np.float32)
    raw[0, tokens] = 0.
    return raw


def test_real_host_ordinary_draws_and_commits_match_with_distinct_runtime_identity():
    core = Candidate()
    a = BoundedReferenceCandidate(core).pipeline(bytes(range(32)), condition='ordinary')
    b = mixture.MixtureCandidate(core).pipeline(bytes(range(32)), condition='ordinary')
    try:
        rngs = [random.Random(12), random.Random(12)]
        for tokens in ([32, 33, 34], [35, 36], [EOS]):
            steps = [p.step(head(tokens), rng.getrandbits) for p, rng in zip((a, b), rngs)]
            assert steps[0].token_id == steps[1].token_id
        ra, rb = a.receipt(), b.receipt()
        assert ra['sampling_records'] == rb['sampling_records']
        assert ra['committed_token_ids'] == rb['committed_token_ids']
        assert ra['runtime']['runtime_profile_sha256'] != rb['runtime']['runtime_profile_sha256']
        assert ra['runtime']['score_namespace_sha256'] == rb['runtime']['score_namespace_sha256']
        assert rb['calibrated'] is False
        assert 'mixture' in rb['runtime']['version']
    finally:
        a.close(); b.close()


def test_unsupported_routes_fail_before_sampling():
    candidate = mixture.MixtureCandidate(Candidate())
    with pytest.raises(ValueError, match='plain generation'):
        candidate.pipeline(bytes(range(32)), condition='marked', allow_tools=True)
    pipe = candidate.pipeline(bytes(range(32)), condition='marked')
    try:
        with pytest.raises(Exception):
            pipe.step(head([THINK_OPEN]), random.Random(1).getrandbits)
        assert pipe.receipt()['final'] is None
    finally:
        pipe.close()


class Backend:
    array = staticmethod(np.array)
    int32, float32 = np.int32, np.float32
    eval = staticmethod(lambda _: None)


@pytest.mark.parametrize('condition', ['ordinary', 'marked'])
def test_research_caller_reference_matches_sdk_receipts(tmp_path, monkeypatch, condition):
    from keyprint import Keyprint
    from keyprint.backends.mlx_bounded import BoundedReferencePublicCandidate
    kp = Keyprint(key=bytes(range(32)))
    kp._candidate = BoundedReferencePublicCandidate(kp._candidate)
    def model(ids, **_):
        return np.repeat(head([32, 33, 34])[:, None, :], ids.shape[1], axis=1)
    monkeypatch.setattr('keyprint.api.secrets.randbits', random.Random(19).getrandbits)
    baseline = kp._run(model, [32, 33, 34], max_tokens=5, condition=condition,
                       output=tmp_path / 'sdk', backend=Backend, cache_factory=lambda _: [])
    research = runner.run(kp._candidate._core, model, [32, 33, 34], key=bytes(range(32)),
                          condition=condition, max_tokens=5, output=tmp_path / 'research',
                          backend=Backend, cache_factory=lambda _: [], random_bits=random.Random(19).getrandbits)
    assert research['text'] == baseline.text
    assert research['receipt']['sampling_records'] == baseline.report['payload']['sampling_records']
    assert research['model_forward_calls'] == 6
    assert research['runtime']['specification']['sdk_execution'] is False


def test_research_caller_handles_changed_profile_and_eos(tmp_path):
    core = mixture.MixtureCandidate(Candidate())
    calls = []
    def model(ids, **_):
        calls.append(1)
        return np.repeat(head([32, 33] if len(calls) == 1 else [EOS])[:, None, :], ids.shape[1], axis=1)
    result = runner.run(core, model, [32], key=bytes(range(32)), condition='marked', max_tokens=4,
                        output=tmp_path / 'mixture', backend=Backend, cache_factory=lambda _: [],
                        random_bits=random.Random(7).getrandbits)
    assert result['completion'] == 'eos'
    assert result['exact_rendering']
    assert result['receipt']['committed_token_ids'][-1] == EOS
    assert len(result['text']) == 1
