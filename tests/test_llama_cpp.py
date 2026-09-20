"""Native boundary and lifecycle contracts; actual GGUF runs are separate."""
import ctypes
import json
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest

from keyprint import Keyprint, KeyprintCancelled, KeyprintError
from keyprint.backends.llama_cpp import LlamaCppModel, _binding, _native_bytes


def copy_bytes(value, buffer, size):
    if len(value) > size:
        return -len(value)
    ctypes.memmove(buffer, value, len(value))
    return len(value)


class Native:
    LLAMA_TOKEN_ATTR_NORMAL = 4
    LLAMA_TOKEN_ATTR_CONTROL = 8
    llama_token = ctypes.c_int32
    pieces = [b"", b"A", b"", b"x" * 81]
    attrs = [8, 4, 8, 4]
    eos = {0}
    llama_vocab_type = staticmethod(lambda _: 2)

    def llama_token_get_attr(self, _, i):
        return self.attrs[i]

    def llama_token_is_eog(self, _, i):
        return i in self.eos

    def llama_token_to_piece(self, _, i, buffer, size, lstrip, special):
        assert lstrip == 0 and special is False
        return copy_bytes(self.pieces[i], buffer, size)

    def llama_detokenize(self, _, tokens, count, buffer, size, remove, unparse):
        assert remove is True and unparse is False
        return copy_bytes(b"".join(self.pieces[i] for i in tokens[:count]), buffer, size)


class Model:
    metadata = {"tokenizer.ggml.model": "gpt2", "tokenizer.ggml.add_space_prefix": "false"}
    _model = SimpleNamespace(vocab="fixture")
    def __init__(self):
        self.resets = self.calls = self.closed = 0
        self.ids = []
        self.head = np.zeros(4, dtype=np.float32)
        self._ctx = SimpleNamespace(get_logits=lambda: self.head.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))

    @property
    def scores(self):
        pytest.fail("unpopulated wrapper score cache must never be read")

    def n_vocab(self):
        return 4

    def n_ctx(self):
        return 128

    def eval(self, ids):
        self.ids.append(list(ids))
        self.calls += 1
        self.head[:] = -np.inf
        self.head[1 if len(self.ids) % 2 else 0] = 0
        self.head[2] = 99  # masked control

    def reset(self):
        self.resets += 1

    def close(self):
        self.closed += 1

    def tokenize(self, text, *, add_bos, special):
        assert add_bos is False
        self.tokenize_options = dict(special=special)
        return [1] if special else [1] * len(text)


def candidate(model=None, native=None):
    model, native = model or Model(), native or Native()
    binding = _binding(model, native)
    formatter = lambda **kw: SimpleNamespace(prompt=kw['messages'][0]['content'])
    kp = Keyprint(key=bytes(range(32)))
    kp._backend = LlamaCppModel(model, native, binding, formatter, temperature=.7, top_k=4)
    return kp


@pytest.mark.parametrize('value', [b'', b'A\0B', b'x' * 81, '日'.encode()])
def test_native_buffer_exact_resize_and_embedded_zero(value):
    assert _native_bytes(lambda buf, n: copy_bytes(value, buf, n)) == value


@pytest.mark.parametrize('value', [-10000, 33])
def test_native_buffer_invalid_lengths(value):
    with pytest.raises(ValueError):
        _native_bytes(lambda *_: value, limit=100)


def test_native_buffer_repeated_resize_is_rejected():
    with pytest.raises(ValueError, match='invalid length'):
        _native_bytes(lambda *_: -80)


def test_binding_preserves_long_pieces_and_all_eog_ids():
    native = Native()
    native.eos = {0, 2}
    binding = _binding(Model(), native)
    assert binding.pieces == (None, b'A', None, b'x' * 81)
    assert binding.eos_ids == {0, 2}


@pytest.mark.parametrize('change', ['normalizer', 'no_eos', 'visible_control', 'ordinary_eos', 'empty', 'non_bpe', 'space'])
def test_unknown_binding_semantics_fail_closed(change):
    model, native = Model(), Native()
    native.attrs, native.pieces, native.eos = list(native.attrs), list(native.pieces), set(native.eos)
    model.metadata = dict(model.metadata)
    if change == 'normalizer': native.attrs[1] = 64
    if change == 'no_eos': native.eos = set()
    if change == 'visible_control': native.pieces[0] = b'eos'
    if change == 'ordinary_eos': native.eos = {1}
    if change == 'empty': native.pieces[1] = b''
    if change == 'non_bpe': model.metadata['tokenizer.ggml.model'] = 'llama'
    if change == 'space': model.metadata['tokenizer.ggml.add_space_prefix'] = 'true'
    with pytest.raises(ValueError):
        _binding(model, native)


@pytest.mark.parametrize('condition', ['ordinary', 'marked'])
def test_native_head_commit_cache_reset_and_exact_decode(condition, tmp_path):
    kp = candidate()
    result = kp.generate('hello', condition=condition, max_tokens=4, output=tmp_path/'run')
    assert result.text == 'A'
    assert result.report['committed_token_ids'] == [1, 0]
    assert kp._backend.model.ids == [[1], [1]]
    assert kp._backend.model.resets == 2
    assert result.report['model_calls'] == 2
    assert result.report['tokenizer_rendering_check'] == 'exact_match'
    assert kp._backend.decode_tokens([3]) == 'x' * 81
    assert kp.inspect('AA').trials >= 0
    assert kp._backend.model.tokenize_options == {'special': False}


def test_raw_head_snapshot_does_not_alias_next_eval():
    backend = candidate()._backend
    with backend.inference_session() as forward:
        first = forward([1])
        saved = first.copy()
        forward([1])
        np.testing.assert_array_equal(first, saved)


def test_failure_resets_cache_and_releases_lock(tmp_path):
    model = Model()
    original = model.eval
    def invalid(ids):
        original(ids)
        model.head[0] = np.nan
    model.eval = invalid
    kp = candidate(model)
    with pytest.raises(KeyprintError) as caught:
        kp.generate('hello', output=tmp_path/'bad')
    assert caught.value.report['model_calls'] == 1
    assert caught.value.report['committed_token_ids'] == []
    assert model.resets == 2
    assert not kp._backend._lock.locked()
    model.eval = original
    model.ids = []
    assert kp.generate('hello', output=tmp_path/'good').text == 'A'


def test_cancellation_after_forward_does_not_sample_and_model_can_be_reused(tmp_path):
    model, stop = Model(), Event()
    original = model.eval
    def stopping(ids):
        original(ids)
        stop.set()
    model.eval = stopping
    kp = candidate(model)
    with pytest.raises(KeyprintCancelled) as caught:
        kp.generate('hello', output=tmp_path/'cancelled', cancel_event=stop)
    assert caught.value.report['committed_token_ids'] == []
    assert model.calls == 1 and model.resets == 2
    records = [json.loads(line)['event'] for line in (tmp_path/'cancelled/journal.jsonl').read_text().splitlines()]
    assert not any(r['phase'] == 'random_requested' for r in records)
    model.eval = original
    model.ids = []
    assert kp.generate('hello', output=tmp_path/'fresh').text == 'A'


def test_busy_generate_score_and_close_are_rejected_before_side_effects(tmp_path):
    kp = candidate()
    with kp._backend._exclusive():
        for call in (lambda: kp.generate('hello', output=tmp_path/'no-run'), lambda: kp.score('A'), kp.close):
            with pytest.raises(RuntimeError, match='busy'):
                call()
    assert not kp._closed and not (tmp_path/'no-run').exists()
    assert kp._backend.model.calls == 0


def test_context_manager_closes_once_and_never_falls_back():
    kp = candidate()
    with kp:
        assert kp.identity['profile'] == 'gguf-byte-bpe-v1-experimental'
    kp.close()
    assert kp._backend.model.closed == 1
    for call in (lambda: kp.generate('hello'), lambda: kp.inspect('A'), kp.pipeline, kp.__enter__):
        with pytest.raises(RuntimeError, match='closed'):
            call()


def test_schema_rejected_before_native_work():
    kp = candidate()
    with pytest.raises(ValueError, match='MLX or Transformers'):
        kp.generate('hello', json_schema={'type': 'object'})
    assert kp._backend.model.calls == kp._backend.model.resets == 0


@pytest.mark.parametrize('settings', [{'threads': 0}, {'threads': True}, {'context_size': 99}, {'context_size': 10000}])
def test_load_validates_options_before_native_allocation(tmp_path, settings):
    with pytest.raises(ValueError):
        LlamaCppModel.load(tmp_path/'not-downloaded', **settings)


def test_cli_pinned_gguf_resolves_file_and_routes_loader(monkeypatch, tmp_path):
    from keyprint import cli
    from keyprint.backends.llama_cpp import MODEL_FILE
    monkeypatch.setattr(cli, 'model_cache_root', lambda: tmp_path)
    repo, revision, files, _ = cli.pinned_model('llama-cpp')
    model = tmp_path/('models--' + repo.replace('/', '--'))/'snapshots'/revision/MODEL_FILE
    with pytest.raises(ValueError, match='No download was started'):
        cli.cached_model('llama-cpp')
    model.parent.mkdir(parents=True)
    model.write_bytes(b'GGUF')
    assert cli.cached_model('llama-cpp') == model
    assert files == [MODEL_FILE]
    assert cli.model_loader('llama-cpp', 'reference', Keyprint) == Keyprint.from_llama_cpp
    with pytest.raises(ValueError, match='MLX option'):
        cli.model_loader('llama-cpp', 'experimental-native', Keyprint)


@pytest.mark.parametrize('command', ['generate', 'serve', 'playground'])
def test_cli_entry_accepts_gguf_file(monkeypatch, tmp_path, command):
    from keyprint import cli
    calls = []
    model = tmp_path/'model.gguf'
    model.write_bytes(b'GGUF')
    key = tmp_path/'key'
    key.write_bytes(bytes(range(32)))
    key.chmod(0o600)
    def loader(path, **kwargs):
        calls.append(path)
        return SimpleNamespace(generate=lambda *a, **kw: SimpleNamespace(text='fixture', artifacts=tmp_path))
    monkeypatch.setattr(Keyprint, 'from_llama_cpp', staticmethod(loader))
    args = [command, '--backend', 'llama-cpp', '--model', str(model), '--key', str(key)]
    if command == 'generate':
        args += ['--prompt', 'hello']
    else:
        pytest.importorskip('fastapi')
        import uvicorn
        import keyprint.server
        import keyprint.playground
        monkeypatch.setattr(uvicorn, 'run', lambda *a, **kw: None)
        monkeypatch.setattr(keyprint.server, 'create_app', lambda fn, **kw: fn())
        monkeypatch.setattr(keyprint.playground, 'create_playground', lambda fn, **kw: fn())
        if command == 'serve':
            token = tmp_path/'api-key'
            token.write_bytes(bytes(reversed(range(32))))
            token.chmod(0o600)
            args += ['--api-key', str(token)]
    assert cli.main(args) == 0
    assert calls == [model]
