import importlib.util
from pathlib import Path

import pytest

from keyprint._engine.legacy._impl.research.grouped_canonical_prototype import Config, Profile

spec = importlib.util.spec_from_file_location("prf_context", Path(__file__).parents[1] / "tools/benchmark_prf_context.py")
candidate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(candidate)


@pytest.mark.parametrize("context", [(), (b"a",), (b"\x00", b"a\x00b", "你好".encode(), b"x" * 256)])
@pytest.mark.parametrize("layers", [1, 30])
def test_exact_prf_addresses_and_cross_key_isolation(context, layers):
    labels = (b"a", b"ab", b"a\x00b", "é".encode(), b"x" * 513)
    profile = Profile(labels, tokenizer_identity="prf-address-test", config=Config(layers=layers))
    for key in (bytes(range(32)), bytes(reversed(range(32))), bytes(range(32))):
        actual = candidate.reused_context_bits(profile, key, context, labels)
        expected = {label: profile.bits(key, context, label) for label in labels}
        assert actual == expected


def test_custom_profile_bits_are_not_silently_replaced():
    calls = []
    class CustomProfile(Profile):
        def bits(self, key, context, label):
            calls.append((key, context, label))
            return (1, 0)
    profile = CustomProfile([b"a", b"b"], tokenizer_identity="fixture")
    result = candidate.reused_context_bits(profile, b"public-key", (b"previous",), [b"a", b"b"])
    assert result == {b"a": (1, 0), b"b": (1, 0)}
    assert calls == [(b"public-key", (b"previous",), b"a"), (b"public-key", (b"previous",), b"b")]
