"""Admission and text-preservation contracts; real inference is separate."""
import hashlib
import json

import pytest

from keyprint import Keyprint
from keyprint.backends.bytelevel import ByteLevelBinding
from keyprint.experimental import wide_mlx
from keyprint.experimental.wide_mlx import NFCWideByteLevelBinding, WideMLXModel
from test_bytelevel import fixture


def test_default_binding_identity_remains_exactly_the_original_profile():
    data = fixture()
    binding = ByteLevelBinding.create(data, vocabulary_size=6, special_ids=[0], eos_ids=[0])
    # Original serialization contract, independent of the new class variables.
    original = json.dumps({"profile": "keyprint-portable-bytelevel-v1-experimental",
        "tokenizer": json.loads(data), "vocabulary_size": 6, "special_ids": [0], "eos_ids": [0],
        "channels": "visible_text_only", "source_policy": "general"}, sort_keys=True, separators=(",", ":"))
    assert binding.digest == hashlib.sha256(original.encode()).hexdigest()
    with pytest.raises(ValueError):
        ByteLevelBinding.create(data, vocabulary_size=248320, special_ids=[0], eos_ids=[0])
    with pytest.raises(ValueError):
        ByteLevelBinding.create(fixture(normalizer={"type": "NFC"}), vocabulary_size=6, special_ids=[0], eos_ids=[0])


@pytest.mark.parametrize("normalizer", [None, {"type": "NFKC"}, {"type": "NFD"},
    {"type": "Sequence", "normalizers": [{"type": "NFC"}]}, {"type": "NFC", "unexpected": True}])
def test_new_profile_only_admits_exact_declared_normalizer(normalizer):
    with pytest.raises(ValueError):
        NFCWideByteLevelBinding.create(fixture(normalizer=normalizer), vocabulary_size=248320, special_ids=[0], eos_ids=[0])


def test_nfc_binding_keeps_generated_bytes_and_refuses_normalized_literal_replay():
    from tokenizers import Tokenizer, decoders, models, normalizers
    tokenizer = Tokenizer(models.BPE({"<eos>": 0, "e": 1, "Ì": 2, "ģ": 3, "é": 4, "Ã": 5, "©": 6}, []))
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.normalizer = normalizers.NFC()
    tokenizer.add_special_tokens(["<eos>"])
    binding = NFCWideByteLevelBinding.create(tokenizer.to_str(), vocabulary_size=248320, special_ids=[0], eos_ids=[0])
    assert binding.render([1, 2, 3]) == "e\u0301"  # never normalize generated text
    assert binding.render([5, 6]) == "é"
    backend = object.__new__(WideMLXModel)
    backend.binding = binding
    backend.encode_literal = lambda text: [5, 6]  # NFC-normalized literal
    with pytest.raises(ValueError, match="not exactly replayable"):
        backend.score("e\u0301", bytes(32))


def test_asset_hashes_and_unexpected_loader_files_are_checked(tmp_path, monkeypatch):
    (tmp_path / "config.json").write_text("{}")
    expected = {"config.json": hashlib.sha256(b"{}").hexdigest()}
    monkeypatch.setattr(wide_mlx, "ASSETS", expected)
    assert wide_mlx.verify_assets(tmp_path) == expected
    (tmp_path / "generation_config.json").write_text("{}")
    with pytest.raises(ValueError, match="unexpected model assets"):
        wide_mlx.verify_assets(tmp_path)
    (tmp_path / "generation_config.json").unlink()
    (tmp_path / "config.json").write_text('{"eos_token_id":0}')
    with pytest.raises(ValueError, match="pinned model asset differs"):
        wide_mlx.verify_assets(tmp_path)


def test_public_opt_in_routes_settings_and_blocks_json_before_inference(monkeypatch, tmp_path):
    seen = []
    backend = object.__new__(WideMLXModel)
    backend.model = object()
    backend.tokenizer = object()
    backend.generate = lambda *a, **k: pytest.fail("unsupported schema reached inference")
    def load(path, **settings):
        seen.append((path, settings))
        return backend
    monkeypatch.setattr(WideMLXModel, "load", load)
    with Keyprint.from_mlx(tmp_path, key=bytes(32), execution="experimental-wide", temperature=.8, top_k=8) as wm:
        assert seen == [(tmp_path, {"temperature": .8, "top_k": 8})]
        with pytest.raises(ValueError, match="json_schema requires"):
            wm.generate("hello", json_schema={"type": "object"})
    assert backend.model is None and backend.tokenizer is None
    with pytest.raises(RuntimeError, match="closed"):
        wm.generate("hello")


def test_bad_settings_rejected_before_asset_reads(monkeypatch, tmp_path):
    monkeypatch.setattr(wide_mlx, "verify_assets", lambda _: pytest.fail("invalid settings touched model assets"))
    with pytest.raises(ValueError):
        WideMLXModel.load(tmp_path, temperature=0)


@pytest.mark.parametrize("condition", ["ordinary", "marked"])
def test_native_request_local_cache_high_token_ids_and_exact_eos(tmp_path, condition):
    mx = pytest.importorskip("mlx.core")
    import numpy as np
    from types import SimpleNamespace
    class Model:
        def __init__(self):
            self.caches, self.inputs = [], []
        def make_cache(self):
            cache = [{"step": 0}]
            self.caches.append(cache)
            return cache
        def __call__(self, ids, *, cache):
            self.inputs.append(ids.tolist())
            step = cache[0]["step"]
            cache[0]["step"] += 1
            raw = np.full((1, 1, 248320), -np.inf, np.float32)
            raw[0, 0, 200000 if step == 0 else 248046] = 0
            raw[0, 0, 248044] = 999  # nested config EOS is NOT the runtime EOS
            return mx.array(raw)
    data = {"model": {"type": "BPE", "vocab": {"A": 200000, "<end>": 248044, "<eos>": 248046}},
            "decoder": {"type": "ByteLevel"}, "normalizer": {"type": "NFC"},
            "added_tokens": [{"id": i, "content": t, "special": True} for i, t in [(248044, "<end>"), (248046, "<eos>")]]}
    binding = NFCWideByteLevelBinding.create(json.dumps(data), vocabulary_size=248320, special_ids=[248044, 248046], eos_ids=[248046])
    model = Model()
    tokenizer = SimpleNamespace(apply_chat_template=lambda *a, **k: [200000],
        decode=lambda ids, **k: binding.render(ids))
    wm = Keyprint(key=bytes(32), top_k=4)
    wm._backend = WideMLXModel(model, tokenizer, binding, {}, top_k=4)
    for n in range(2):
        result = wm.generate("hello", condition=condition, max_tokens=4, output=tmp_path / str(n), trace=True)
        assert result.text == "A"
        assert result.report["committed_token_ids"] == [200000, 248046]
        assert result.report["completion"] == "eos"
        assert result.report["model_calls"] == 2
        assert "".join(t.text for t in result.trace) == "A"
    assert model.caches[0] is not model.caches[1]
    assert model.caches == [[], []]  # released even while model stays loaded
    assert model.inputs == [[[200000]]] * 4
    wm.close()
