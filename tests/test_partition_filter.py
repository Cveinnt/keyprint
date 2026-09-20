"""Full filtering parity, including validation and rounded gap boundaries."""
import numpy as np
import pytest
import sys

from keyprint.experimental.partition_filter import partition_support_filter
from keyprint._engine.research.keyprint_stable_support_filter_v3 import stable_support_filter


@pytest.fixture(params=[False, True], autouse=True)
def compare_backend(request, monkeypatch):
    if request.param:
        native = pytest.importorskip('keyprint_native').NativePRF()
        original = partition_support_filter
        monkeypatch.setattr(sys.modules[__name__], 'partition_support_filter',
                            lambda *a, **k: original(*a, **k, native=native))


def compare(head, **settings):
    original = head.tobytes() if isinstance(head, np.ndarray) else None
    try:
        expected = stable_support_filter(head, **settings)
    except (ValueError, TypeError, ArithmeticError) as exc:
        with pytest.raises(type(exc)) as caught:
            partition_support_filter(head, **settings)
        assert str(caught.value) == str(exc)
        return
    actual = partition_support_filter(head, **settings)
    assert actual.filtered_logits.tobytes() == expected.filtered_logits.tobytes()
    assert actual.filtered_logits.dtype == expected.filtered_logits.dtype
    assert actual.admitted_token_ids == expected.admitted_token_ids
    assert actual.identity == expected.identity
    assert actual.diagnostics == expected.diagnostics
    assert not actual.filtered_logits.flags.writeable
    assert head.tobytes() == original


@pytest.mark.parametrize("width", [1, 100, 101, 2048, 151936])
@pytest.mark.parametrize("temperature,top_k,gap", [(.7, 100, 600.), (5e-324, 10, 600.),
    (1e308, 100, 600.), (1., 1, 1e-300), (1., 151669, 600.)])
def test_full_numeric_contract_matches_reference(width, temperature, top_k, gap):
    rng = np.random.default_rng(1000 + width)
    random = (rng.normal(size=(1, width)) * 100).astype(np.float32)
    ties = rng.integers(-4, 5, size=(1, width)).astype(np.float32)
    extremes = np.resize(np.array([-np.finfo(np.float32).max, np.finfo(np.float32).max,
        -0., 0., np.nextafter(np.float32(0), np.float32(1)), -np.inf], dtype=np.float32), (1, width))
    for row in (random, ties, extremes, np.zeros((1, width), dtype=np.float32)):
        compare(row, temperature=temperature, top_k=top_k, max_logit_gap=gap)


@pytest.mark.parametrize("temperature", [.1, .7, 1., 10.])
def test_gap_neighbors_and_unmapped_padding(temperature):
    boundary = np.float32(-600 * temperature)
    head = np.array([[0., boundary, np.nextafter(boundary, np.float32(-np.inf)),
                      np.nextafter(boundary, np.float32(np.inf)), 1000000.]], dtype=np.float32)
    compare(head, temperature=temperature, mapped_vocabulary_size=4)


@pytest.mark.parametrize("head,settings", [
    ([[1]], {}), (np.zeros((1, 4), dtype=np.float64), {}),
    (np.zeros((2, 4), dtype=np.float32), {}), (np.zeros((1, 0), dtype=np.float32), {}),
    (np.array([[0, np.nan]], dtype=np.float32), {}),
    (np.array([[0, np.inf]], dtype=np.float32), {"mapped_vocabulary_size": 1}),
    (np.array([[-np.inf]], dtype=np.float32), {}),
    (np.zeros((1, 4), dtype=np.float32), {"temperature": 0}),
    (np.zeros((1, 4), dtype=np.float32), {"temperature": True}),
    (np.zeros((1, 4), dtype=np.float32), {"temperature": float('inf')}),
    (np.zeros((1, 4), dtype=np.float32), {"top_k": True}),
    (np.zeros((1, 4), dtype=np.float32), {"top_k": 0}),
    (np.zeros((1, 4), dtype=np.float32), {"top_k": 151670}),
    (np.zeros((1, 4), dtype=np.float32), {"max_logit_gap": 601}),
    (np.zeros((1, 4), dtype=np.float32), {"mapped_vocabulary_size": 5}),
])
def test_invalid_input_and_settings_fail_identically(head, settings):
    compare(head, **settings)
