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


def test_ascii_added_tokens_match_real_decoder_with_adjacent_fragment_bytes():
    from tokenizers import AddedToken, Tokenizer, decoders, models
    tokenizer = Tokenizer(models.BPE({"<eos>": 0, "A": 1, "Ġ": 2, "Ã": 3, "©": 4}, []))
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.add_special_tokens(["<eos>"])
    tokenizer.add_tokens([AddedToken("<think>", normalized=False),
                          AddedToken("</think>", normalized=False)])
    binding = ByteLevelBinding.create(tokenizer.to_str(), vocabulary_size=7,
                                      special_ids=[0], eos_ids=[0])
    assert binding.pieces[5:] == (b"<think>", b"</think>")
    for ids in ([5], [6], [1, 5, 3, 4, 6, 2, 1, 0], [5, 5, 6]):
        assert binding.render(ids) == tokenizer.decode(ids, skip_special_tokens=True)
    assert binding.render([1, 5, 3, 4, 6, 2, 1, 0]) == "A<think>é</think> A"


@pytest.mark.parametrize("change", [
    {"content": "é"}, {"content": " "}, {"content": ""}, {"content": "a b"},
    {"single_word": True}, {"lstrip": True}, {"rstrip": True}, {"normalized": True},
    {"id": 0}, {"id": True}, {"id": 6}, {"content": "A"},
])
def test_ambiguous_added_token_rejected(change):
    added = {"id": 5, "content": "<think>", "special": False, "normalized": False, **change}
    data = json.loads(fixture())
    data["added_tokens"].append(added)
    with pytest.raises(ValueError):
        ByteLevelBinding.create(json.dumps(data), vocabulary_size=6, special_ids=[0], eos_ids=[0])


def test_added_token_cannot_alias_an_existing_id():
    data = json.loads(fixture())
    data["added_tokens"].append({"id": 1, "content": "<think>", "special": False})
    with pytest.raises(ValueError, match="duplicate token IDs"):
        ByteLevelBinding.create(json.dumps(data), vocabulary_size=6, special_ids=[0], eos_ids=[0])
