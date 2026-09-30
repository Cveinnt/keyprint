"""Backend-owned tokenization must not allocate the unrelated Qwen reference."""
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from keyprint import Keyprint
from keyprint._engine.research import keyprint_v3_public_api_rc2 as reference


@pytest.mark.parametrize('backend', ['transformers', 'llama_cpp', 'wide_mlx'])
def test_backend_factory_and_inspection_do_not_construct_reference(monkeypatch, tmp_path, backend):
    def forbidden(**settings):
        pytest.fail('unrelated reference tokenizer allocated')
    monkeypatch.setattr(reference, 'PublicCandidate', forbidden)
    report = dict(kind='literal_diagnostic', eligible_events=0, one_bits=0, total_bits=0)
    native = SimpleNamespace(identity={'fixture': backend}, score=lambda *args: report, close=lambda: None)
    if backend == 'transformers':
        from keyprint.backends.transformers import TransformersModel
        monkeypatch.setattr(TransformersModel, 'load', lambda *a, **kw: native)
        sdk = Keyprint.from_transformers(tmp_path, key=bytes(32))
    elif backend == 'llama_cpp':
        from keyprint.backends.llama_cpp import LlamaCppModel
        monkeypatch.setattr(LlamaCppModel, 'load', lambda *a, **kw: native)
        sdk = Keyprint.from_llama_cpp(tmp_path, key=bytes(32))
    else:
        from keyprint.experimental.wide_mlx import WideMLXModel
        monkeypatch.setattr(WideMLXModel, 'load', lambda *a, **kw: native)
        sdk = Keyprint.from_mlx(tmp_path, key=bytes(32), execution='experimental-wide')
    assert sdk.identity == native.identity
    assert sdk.inspect('original text').trials == 0
    with pytest.raises(ValueError, match='reference binding'):
        sdk.pipeline()
    sdk.close()


def test_reference_initializes_once_with_original_settings(monkeypatch):
    calls = []
    target = SimpleNamespace(core_identity={'fixture': 'reference'})
    def create(**settings):
        calls.append(settings)
        return target
    monkeypatch.setattr(reference, 'PublicCandidate', create)
    sdk = Keyprint(key=bytes(32), temperature=.5, top_k=7)
    assert calls == []
    assert sdk.identity == sdk.identity == target.core_identity
    assert calls == [dict(temperature=.5, top_k=7)]


@pytest.mark.parametrize('settings', [dict(temperature=0), dict(temperature=float('nan')),
    dict(temperature=True), dict(top_k=0), dict(top_k=True)])
def test_invalid_policy_still_fails_at_construction(monkeypatch, settings):
    monkeypatch.setattr(reference, 'PublicCandidate', lambda **kw: pytest.fail('allocated'))
    with pytest.raises(ValueError):
        Keyprint(key=bytes(32), **settings)


def test_lazy_allocation_cannot_transfer_reference_thread_ownership(monkeypatch):
    monkeypatch.setattr(reference, 'PublicCandidate', lambda **kw: pytest.fail('allocated'))
    sdk = Keyprint(key=bytes(32))
    with ThreadPoolExecutor(max_workers=1) as executor:
        with pytest.raises(RuntimeError, match='another thread'):
            executor.submit(lambda: sdk.identity).result()


def test_closed_instance_cannot_allocate_reference(monkeypatch):
    monkeypatch.setattr(reference, 'PublicCandidate', lambda **kw: pytest.fail('allocated'))
    sdk = Keyprint(key=bytes(32))
    sdk.close()
    with pytest.raises(RuntimeError, match='closed'):
        sdk.pipeline()
