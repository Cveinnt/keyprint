"""Study rules must not silently tune on held-out data or drop failed attempts."""
import importlib.util
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location("detection_screen", Path(__file__).parents[1] / "tools/validate_detection_screen.py")
screen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(screen)


def test_threshold_rejects_heldout_and_failed_calibration():
    row = {"split": "calibration", "condition": "ordinary", "scores": {"matching": {"statistic": 2.5}}}
    assert screen.freeze_threshold([row]) == 2.5
    for bad in ({**row, "split": "heldout"}, {**row, "condition": "marked"}, {**row, "error": "ValueError"}):
        with pytest.raises(ValueError):
            screen.freeze_threshold([row, bad])


def test_unavailable_attempts_retained_and_ties_not_positive():
    rows = [{"split": "heldout", "condition": "marked", "scores": {
        "matching": {"statistic": 2.5}, "other": {"statistic": 3.0}}},
        {"split": "heldout", "condition": "marked", "error": "ValueError"}]
    result = screen.summarize(rows, 2.5)
    assert result["marked_matching"]["above_frozen_threshold"] == 0
    assert result["marked_matching"]["attempts"] == 2
    assert result["marked_matching"]["unavailable"] == 1
    assert result["marked_other"]["above_frozen_threshold"] == 1


@pytest.mark.parametrize("ones,trials", [(0, 0), (5, 4), (-1, 5), (True, 5), (1, 2.0)])
def test_missing_or_invalid_counts_cannot_be_zero_signal(ones, trials):
    with pytest.raises(ValueError):
        screen.statistic(ones, trials)
