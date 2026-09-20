import pytest

from keyprint.backends.bytelevel import ByteLevelBinding
from keyprint.experimental.completion import finalize_completion


@pytest.fixture
def binding():
    return ByteLevelBinding((None, b"A", b"\xc3", b"\xa9", b"\xef\xbf\xbd", None,
                             b"\xf0\x9f", b"\x8c\xb1"), frozenset({0}), "fixture")


def finish(binding, ids, text, reason="length", cap=None):
    return finalize_completion(binding, token_ids=ids, text=text, finish_reason=reason,
                               max_tokens=len(ids) if cap is None else cap)


@pytest.mark.parametrize("ids,text,pending", [
    ([1, 2], "A", b"\xc3"), ([1, 2], "A\ufffd", b"\xc3"),
    ([2], "", b"\xc3"), ([6], "\ufffd", b"\xf0\x9f"),
    ([1, 2, 3], "Aé", b""), ([6, 7], "🌱", b""),
    ([4], "\ufffd", b""), ([4, 2], "\ufffd\ufffd", b"\xc3"),
])
def test_cap_preserves_all_bytes_without_extra_tokens(binding, ids, text, pending):
    result = finish(binding, ids, text)
    assert result.text.encode() + result.pending_utf8 == b"".join(binding.pieces[i] for i in ids)
    assert result.pending_utf8 == pending
    assert result.token_ids == tuple(ids)
    assert result.host_text == text
    assert result.completion == "length"
    assert result.carrier_rendering["host_text_adjusted"] == (result.text != text)


@pytest.mark.parametrize("reason,cap", [("stop", 8), ("stop", 2), ("length", 2)])
def test_eos_and_cap_collision_preserve_host_reason(binding, reason, cap):
    result = finish(binding, [1, 0], "A", reason, cap)
    assert result.completion == ("eos" if reason == "stop" else "length")
    assert result.pending_utf8 == b""


@pytest.mark.parametrize("ids,text,reason,cap", [
    ([1], "A", "abort", 1), ([1], "A", "error", 1), ([1], "A", None, 1),
    ([1], "A", "stop", 2), ([1], "A", "length", 2),
    ([0, 1], "A", "length", 2), ([5], "", "length", 1),
    ([True], "A", "length", 1), ([-1], "", "length", 1),
    ([8], "", "length", 1), ([], "", "length", 1),
    ([1], "A", "length", True), ([1], "A", "length", 1025),
    ([1, 2], "B\ufffd", "length", 2), ([1, 2], "A\ufffd\ufffd", "length", 2),
    ([1, 2], "", "length", 2), ([1], " A", "length", 1),
    ([4], "", "length", 1), ([1], None, "length", 1),
])
def test_unsupported_or_inconsistent_host_output_fails(binding, ids, text, reason, cap):
    with pytest.raises(ValueError):
        finish(binding, ids, text, reason, cap)


@pytest.mark.parametrize("ids,text,reason", [
    ([2, 0], "\ufffd", "stop"), ([2, 0], "\ufffd", "length"),
    ([3], "\ufffd", "length"), ([2, 1], "\ufffdA", "length"),
])
def test_malformed_utf8_or_incomplete_eos_is_not_repaired(binding, ids, text, reason):
    with pytest.raises(UnicodeDecodeError):
        finish(binding, ids, text, reason)
