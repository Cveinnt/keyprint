import hmac
from pathlib import Path
import sys

import pytest
from keyprint.experimental.hmac_context import SHAContext

TOOLS = Path(__file__).parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
try:
    import benchmark_sha_context as candidate
finally:
    sys.path.remove(str(TOOLS))


@pytest.mark.parametrize("key", [bytes(32),bytes(range(32)),bytes([255])*32])
@pytest.mark.parametrize("size", [0,1,55,56,63,64,65,127,128,1024])
def test_entire_digest_matches_stdlib_at_sha_block_boundaries(key,size):
    prefix = bytes(range(256)) * 2
    suffix = b"x" * size + "你好".encode()
    context = SHAContext(key,prefix,30)
    for layer in range(30):
        expected = hmac.digest(key,prefix+layer.to_bytes(4,"big")+suffix,"sha256")
        assert context.digest(layer,suffix)==expected


def test_address_bits_and_independent_keys_contexts_match():
    labels = [b"a",b"a\x00b","é".encode(),b"x"*513]
    p = candidate.Profile(labels,tokenizer_identity="test",config=candidate.Config(layers=30))
    for key in (bytes(range(32)),bytes(reversed(range(32))),bytes(range(32))):
        for context in ((),(b"a",),(b"\x00",b"a\x00b","你好".encode(),b"x"*256)):
            assert candidate.sha_context_bits(p,key,context,labels)=={label:p.bits(key,context,label) for label in labels}


@pytest.mark.parametrize("key", [b"",b"x"*31,b"x"*33,"x"*32])
def test_invalid_keys_rejected(key):
    with pytest.raises(ValueError): SHAContext(key,b"",30)
