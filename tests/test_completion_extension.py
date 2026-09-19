import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest


@pytest.fixture
def study(monkeypatch):
    tools = Path(__file__).parents[1] / 'tools'
    for name in ('compare_score_baselines', 'validate_null_corpus', 'analyze_mlx_capacity', 'validate_completion_extension'):
        spec = importlib.util.spec_from_file_location(name, tools / (name + '.py'))
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
    return module


def saved(study, fresh):
    logits = np.array([[1., 2.]], dtype=np.float32)
    events = [dict(kind='prepared_step', index=0, raw_logits_sha256=hashlib.sha256(logits.tobytes()).hexdigest()),
              dict(kind='random_bits_returned', sample_index=0, bits=8, value_decimal='17')]
    return study.PrefixReplay(events, [1], [{'selected': 1, 'probability_hash': 'original'}], fresh), logits


def test_no_new_randomness_until_full_prefix_verified(study):
    calls = []
    replay, logits = saved(study, lambda n: calls.append(n) or 9)
    replay.prepare(logits)
    assert replay.bits(8) == 17 and calls == []
    replay.commit(1, {'selected': 1, 'probability_hash': 'original'})
    replay.prepare(logits)
    assert replay.bits(8) == 9 and calls == [8]
    assert replay.replayed_draws == 1 and replay.new_draws == 1


@pytest.mark.parametrize('fault', ['logits', 'bits', 'token', 'probability', 'extra_draw'])
def test_prefix_mismatch_cannot_consume_fresh_randomness(study, fault):
    calls = []
    replay, logits = saved(study, lambda n: calls.append(n) or 9)
    with pytest.raises(ValueError):
        replay.prepare(logits + 1 if fault == 'logits' else logits)
        replay.bits(7 if fault == 'bits' else 8)
        if fault == 'extra_draw':
            replay.bits(8)
        record = {'selected': 1, 'probability_hash': 'changed' if fault == 'probability' else 'original'}
        replay.commit(0 if fault == 'token' else 1, record)
    assert calls == []


def test_unprepared_randomness_and_partial_commit_rejected(study):
    replay, logits = saved(study, lambda n: 0)
    with pytest.raises(ValueError, match='preparation'):
        replay.bits(8)
    with pytest.raises(ValueError, match='Unprepared'):
        replay.commit(1, {})


def test_public_identity_projection_requires_matching_verified_specification(study):
    candidate = dict(specification={'setting': 3}, version='test', score_namespace_sha256='score',
                     max_steps=2048, deployment_calibrated=False)
    candidate['runtime_profile_sha256'] = hashlib.sha256(json.dumps(candidate['specification'],
        sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    target = {k: v for k, v in candidate.items() if k != 'specification'}
    target['specification_digest_verified'] = True
    assert study.same_identity(candidate, target)
    assert not study.same_identity(candidate, {**target, 'max_steps': 1024})
    assert not study.same_identity(candidate, {**target, 'specification_digest_verified': False})
    assert not study.same_identity({**candidate, 'specification': {'setting': 4}}, target)
