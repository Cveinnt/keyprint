"""Default MLX finalization, with controlled heads and unchanged reference sampling."""
import copy
import random
from threading import Event
from types import SimpleNamespace

import pytest

from keyprint import Keyprint, KeyprintCancelled, KeyprintError
from keyprint.backends.mlx_bounded import BoundedReferencePublicCandidate, VERSION
from keyprint._engine.research.keyprint_candidate_v3.adapter import V3Host
from keyprint._engine.legacy._impl.research.token_channel_host import EOS
from keyprint.experimental.fast_reporting import _experimental_target, _digest
from test_fast_caller import Backend, head
from test_fast_mlx import head as pipeline_head


def candidate():
    sdk = Keyprint(key=bytes(range(32)))
    sdk._candidate = BoundedReferencePublicCandidate(sdk._candidate)
    return sdk


def test_default_model_loading_selects_bound_reference_host(monkeypatch):
    monkeypatch.setattr('keyprint.backends.mlx.MLXModel.load', lambda _: SimpleNamespace())
    sdk = Keyprint.from_mlx('fixture', key=bytes(range(32)))
    assert sdk.identity['version'] == VERSION
    assert sdk.identity != Keyprint(key=bytes(range(32))).identity
    with sdk._candidate._core.pipeline(bytes(range(32)), condition='marked') as pipeline:
        assert type(pipeline._raw) is V3Host


@pytest.mark.parametrize('condition', ['ordinary', 'marked'])
def test_complete_reference_probabilities_draws_text_and_scores_unchanged(condition, tmp_path, monkeypatch):
    reports = []
    for name, sdk in [('frozen', Keyprint(key=bytes(range(32)))), ('bounded', candidate())]:
        monkeypatch.setattr('keyprint.api.secrets.randbits', random.Random(17).getrandbits)
        result = sdk._run(lambda ids, **_: head(ids), [32], max_tokens=5,
            condition=condition, output=tmp_path / name, backend=Backend, cache_factory=lambda _: [])
        reports.append(result)
    a, b = reports
    assert a.text == b.text
    for field in ['committed_token_ids', 'sampling_records', 'completion', 'literal_diagnostics']:
        assert a.report['payload'][field] == b.report['payload'][field]
    assert a.report['usage'] == b.report['usage']
    assert b.report['schema'] == 'keyprint.bounded-reference-report.v1'
    assert a.report['target_identity'] != b.report['target_identity']
    assert candidate().inspect(b.text).fraction == Keyprint(key=bytes(range(32))).inspect(a.text).fraction


@pytest.mark.parametrize('condition', ['ordinary', 'marked'])
@pytest.mark.parametrize('token,pending', [(126, 'c2'), (37698, 'e29c'), (92173, 'f09f92'), (25521, 'e29c')])
def test_exact_cap_retains_bytes_without_sampling_another_token(condition, token, pending, tmp_path):
    sdk = candidate()
    calls = []
    def model(ids, **_):
        calls.append(1)
        return head(ids, (32 if len(calls) <= 2 else token,))
    result = sdk._run(model, [32, 32], max_tokens=2, condition=condition,
                     output=tmp_path / 'run', backend=Backend, cache_factory=lambda _: [])
    assert len(calls) == 3
    assert result.text == ('A ' if token == 25521 else 'A')
    assert result.report['usage']['completion_tokens'] == 2
    assert result.report['payload']['committed_token_ids'] == [32, token]
    assert result.report['payload']['completion'] == 'length'
    assert result.report['carrier_rendering'][0]['pending_utf8_hex'] == pending
    assert result.report['literal_replay_status'][0]['availability'] == 'unavailable'
    assert result.report['payload']['literal_diagnostics'][0]['events'] is None


@pytest.mark.parametrize('ending', [None, EOS])
def test_manual_finish_and_eos_do_not_hide_incomplete_utf8(ending):
    with candidate()._candidate._core.pipeline(bytes(range(32)), condition='ordinary') as pipeline:
        pipeline.step(pipeline_head([25521]), lambda _: 0)
        with pytest.raises(UnicodeDecodeError):
            if ending is None:
                pipeline.finish()
            else:
                pipeline.step(pipeline_head([ending]), lambda _: 0)


def test_cancellation_retains_work_and_model_remains_reusable(tmp_path):
    sdk = candidate(); stop = Event(); calls = []
    def model(ids, **_):
        calls.append(1)
        if len(calls) == 2:
            stop.set()
        return head(ids, (32,))
    with pytest.raises(KeyprintCancelled) as caught:
        sdk._run(model, [32], max_tokens=4, condition='marked', output=tmp_path/'cancelled',
                 backend=Backend, cache_factory=lambda _: [], cancel_event=stop)
    assert caught.value.report['usage']['completion_tokens'] == 1
    result = sdk._run(lambda ids, **_: head(ids, (32,)), [32], max_tokens=1, condition='marked',
                      output=tmp_path/'next', backend=Backend, cache_factory=lambda _: [])
    assert result.text == 'A'


@pytest.mark.parametrize('field', ['sources', 'sampling', 'length_finalization'])
def test_recomputed_digest_cannot_hide_a_changed_execution_contract(field):
    identity = copy.deepcopy(candidate().identity)
    identity['specification']['execution'][field] = 'changed'
    identity['runtime_profile_sha256'] = _digest(identity['specification'])
    with pytest.raises(ValueError, match='execution source mismatch'):
        _experimental_target(identity)
