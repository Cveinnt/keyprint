import pytest

from keyprint import Inspection, Keyprint


def test_inspection_preserves_reference_report_and_key_control():
    key, other = bytes(range(32)), bytes(reversed(range(32)))
    text = "A small library should make a useful capability easy to understand and reproduce."
    wm = Keyprint(key=key)
    result = wm.inspect(text)
    assert result.report == wm.score(text)
    assert result.events == result.report["payload"]["events"]
    assert result.fraction == result.ones / result.trials
    assert wm.inspect(text, key=other).report == Keyprint(key=other).score(text)
    assert result.report["verdict"] is None
    with pytest.raises(ValueError, match="32 bytes"):
        wm.inspect(text, key=b"bad")


def test_zero_and_unavailable_are_not_fifty_percent_or_negative_verdicts():
    zero = Inspection.from_report({"kind": "literal_diagnostic", "eligible_events": 0, "one_bits": 0, "total_bits": 0})
    assert zero.fraction is None and zero.events == 0
    unavailable = Inspection.from_report({"kind": "literal_diagnostic", "payload": {"events": None, "ones": None, "trials": None}})
    assert unavailable.fraction is None and unavailable.events is None
    with pytest.raises(ValueError):
        Inspection.from_report({"kind": "literal_diagnostic", "eligible_events": 1, "one_bits": 40, "total_bits": 30})


def test_inspect_rejects_oversized_input_before_engine():
    with pytest.raises(ValueError, match="16000"):
        Keyprint(key=bytes(range(32))).inspect("x" * 16001)
