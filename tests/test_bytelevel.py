import json
import pytest

from keyprint.backends.bytelevel import ByteLevelBinding


def fixture(**changes):
    data = {"model": {"type": "BPE", "vocab": {"<eos>": 0, "A": 1, "Ġ": 2, "Ã": 3, "©": 4}},
            "decoder": {"type": "ByteLevel"}, "normalizer": None,
            "added_tokens": [{"id": 0, "content": "<eos>", "special": True}]}
    data.update(changes)
    return json.dumps(data)


def test_fragment_bytes_are_preserved_and_eos_excluded():
    binding = ByteLevelBinding.create(fixture(), vocabulary_size=6, special_ids=[0], eos_ids=[0])
    assert binding.pieces == (None, b"A", b" ", b"\xc3", b"\xa9", None)
    assert binding.render([1, 2, 3, 4, 0]) == "A é"
    with pytest.raises(UnicodeDecodeError):
        binding.render([3])


@pytest.mark.parametrize("change", [{"decoder": {"type": "Metaspace"}}, {"normalizer": {"type": "NFC"}},
    {"added_tokens": [{"id": 0, "content": "x", "special": False}]},
    {"model": {"type": "BPE", "vocab": {"bad": 10}}}])
def test_unsupported_binding_rejected(change):
    with pytest.raises(ValueError):
        ByteLevelBinding.create(fixture(**change), vocabulary_size=6, special_ids=[0], eos_ids=[0])


def test_eos_required_and_bound_to_identity():
    with pytest.raises(ValueError):
        ByteLevelBinding.create(fixture(), vocabulary_size=6, special_ids=[], eos_ids=[])
    a = ByteLevelBinding.create(fixture(), vocabulary_size=6, special_ids=[0], eos_ids=[0])
    b = ByteLevelBinding.create(fixture(), vocabulary_size=6, special_ids=[0, 1], eos_ids=[0, 1])
    assert a.digest != b.digest
