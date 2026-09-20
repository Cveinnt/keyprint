from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "tools"))
try:
    from partition_topk import select
finally:
    sys.path.pop(0)


@pytest.mark.parametrize("width", [1, 2, 10, 99, 100, 101, 1024, 151669])
@pytest.mark.parametrize("kind", ["random", "ties", "equal", "signed_zero", "extremes"])
def test_full_order_exactly_matches_reference_including_cutoff_ties(width, kind):
    rng = np.random.default_rng(42 + width)
    ids = np.arange(width, dtype=np.int64) * 2 + 3
    if kind == "random": scores = rng.normal(size=width).astype(np.float32).astype(np.float64)
    elif kind == "ties": scores = rng.integers(-3, 4, size=width).astype(np.float64)
    elif kind == "equal": scores = np.ones(width, dtype=np.float64)
    elif kind == "signed_zero": scores = np.resize(np.array([-0., 0.]), width)
    else:
        scores = np.resize(np.array([-np.finfo(np.float32).max, -np.finfo(np.float32).tiny,
                                     0., np.finfo(np.float32).tiny, np.finfo(np.float32).max], dtype=np.float64), width)
    original = scores.copy()
    for k in sorted({1, min(100, width), width, width + 1}):
        assert np.array_equal(select(ids, scores, k), np.lexsort((ids, -scores))[:k])
    assert scores.tobytes() == original.tobytes()
