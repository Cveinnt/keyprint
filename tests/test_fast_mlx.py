import random

import numpy as np
import pytest

from keyprint.experimental.fast_mlx import pipeline, _ContextProfile
from keyprint._engine.research.keyprint_candidate_v3 import Candidate
from keyprint._engine.legacy._impl.research.token_channel_host import EOS, THINK_OPEN, THINK_CLOSE, TOOL_OPEN, TOOL_CLOSE


def head(tokens, values=None):
    result = np.full((1, 151936), -np.inf, dtype=np.float32)
    result[0, tokens] = 0 if values is None else values
    return result


def pair(condition="marked", **settings):
    key = bytes(range(32))
    return Candidate().pipeline(key, condition=condition, **settings), pipeline(key, condition=condition, **settings)


def assert_parity(reference, fast):
    a, b = reference.receipt(), fast.receipt()
    for field in ("committed_token_ids", "sampling_records", "sampling_attempts", "shared_filter_attempts",
                  "attempted_bit_requests", "completed_bit_returns", "closed", "finalized", "calibrated"):
        assert a[field] == b[field], field
    assert a["runtime"]["runtime_profile_sha256"] != b["runtime"]["runtime_profile_sha256"]
    assert a["runtime"]["score_namespace_sha256"] == b["runtime"]["score_namespace_sha256"]


@pytest.mark.parametrize("condition", ["ordinary", "marked"])
def test_probabilities_tokens_rejections_and_repeated_contexts_match(condition):
    a, b = pair(condition)
    ra, rb = random.Random(12), random.Random(12)
    rng = np.random.default_rng(17)
    try:
        for i in range(12):
            raw = head(list(range(32, 95)), rng.uniform(-40, 0, 63)) if i < 6 else head([32])
            assert a.step(raw, ra.getrandbits).token_id == b.step(raw, rb.getrandbits).token_id
        a.finish(); b.finish()
        assert_parity(a, b)
        assert all(not carrier.session.profile._state for carrier in b._raw.carriers)
    finally:
        a.close(); b.close()


def test_rejected_draw_is_preserved_before_the_same_selected_token():
    a, b = pair("ordinary")
    try:
        for item in (a, b):
            calls = []
            def draw(bits):
                calls.append(bits)
                return (1 << bits) - 1 if len(calls) == 1 else 0
            item.step(head([32, 33, 34]), draw)
            item.finish()
        assert_parity(a, b)
        transcript = a.receipt()["sampling_records"][0]["randomness"]["transcript"]
        assert [d["accepted"] for d in transcript] == [False, True]
    finally:
        a.close(); b.close()


def test_reasoning_and_tool_carriers_have_isolated_context_and_same_routing():
    a, b = pair(allow_thinking=True, allow_tools=True)
    try:
        for token in (THINK_OPEN, 32, THINK_CLOSE, TOOL_OPEN, 33, TOOL_CLOSE, 34, EOS):
            for item in (a, b): item.step(head([token]), random.Random(1).getrandbits)
        assert_parity(a, b)
        assert len({id(c.session.profile._state) for c in b._raw.carriers}) == 3
        assert all(not c.session.profile._state for c in b._raw.carriers)
        assert a.finish().visible.text == b.finish().visible.text
    finally:
        a.close(); b.close()


@pytest.mark.parametrize("failure", ["invalid_bits", "invalid_head", "illegal_control"])
def test_failure_and_consumed_work_match_and_clear_context(failure):
    a, b = pair()
    try:
        for item in (a, b): item.step(head([32, 33]), random.Random(1).getrandbits)
        raw = head([32, 33])
        draw = (lambda _: True) if failure == "invalid_bits" else random.Random(1).getrandbits
        if failure == "invalid_head": raw[0, 32] = np.nan
        if failure == "illegal_control": raw = head([THINK_OPEN])
        exceptions = []
        for item in (a, b):
            with pytest.raises(Exception) as caught: item.step(raw, draw)
            exceptions.append(type(caught.value))
        assert exceptions[0] is exceptions[1]
        assert_parity(a, b)
        assert all(not c.session.profile._state for c in b._raw.carriers)
    finally:
        a.close(); b.close()


def test_proofreading_protection_and_profile_namespace_are_preserved():
    a, b = pair(purpose="proofread", source_text="A" * 128)
    try:
        for _ in range(45):
            for item in (a, b): item.step(head([32]), random.Random(1).getrandbits)
        for item in (a, b): item.step(head([32, 33]), random.Random(1).getrandbits)
        assert a._raw._visible.session.source_receipt() == b._raw._visible.session.source_receipt()
        assert b._raw._visible.session.source_receipt()["decision_counts"]["source_protected"] > 0
        assert a._raw.binding.profile.identity_receipt() == b._raw._visible.session.profile.identity_receipt()
        a.finish(); b.finish()
        assert_parity(a, b)
    finally:
        a.close(); b.close()


def test_context_state_is_local_and_invalidates_on_key_change():
    reference = Candidate()._base._binding.profile
    first, second = _ContextProfile(reference), _ContextProfile(reference)
    for key in (bytes(range(32)), bytes(reversed(range(32))), bytes(range(32))):
        assert first.bits(key, (b"prior",), b"label") == reference.bits(key, (b"prior",), b"label")
    assert first._state and not second._state
    assert first._domain == reference._domain
