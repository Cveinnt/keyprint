"""Full filtering parity, including validation and rounded gap boundaries."""
import numpy as np
import pytest
import sys
import warnings

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


@pytest.mark.parametrize('seed', range(30))
def test_range_certificate_and_fallback_match_random_bits(seed):
    rng = np.random.default_rng(seed)
    width = (3, 17, 256, 4096, 151936)[seed % 5]
    bits = rng.integers(0, 2**32, size=(1, width), dtype=np.uint32)
    # Arbitrary finite binary32 patterns, including subnormals and signed zeros.
    bits[(bits & 0x7f800000) == 0x7f800000] = 0
    head = bits.view(np.float32)
    head[0, ::7] = -np.inf
    head[0, 1] = 0.
    compare(head, temperature=(.7, 1e-200, 1e200)[seed % 3],
            top_k=(1, 100, 512, 151669)[seed % 4],
            max_logit_gap=(600., .5, 1e-100)[seed % 3])


@pytest.mark.parametrize('temperature', [1e308, 1e-308, .7])
def test_caller_underflow_policy_preserved(temperature):
    tiny = np.nextafter(np.float32(0), np.float32(1))
    head = np.array([[0., -tiny, -1., -np.inf]], dtype=np.float32)
    with np.errstate(all='raise'):
        compare(head, temperature=temperature, top_k=1)


@pytest.mark.parametrize('direction', [-np.inf, np.inf])
def test_span_certificate_neighbors_keep_exact_gap_boundary(direction):
    temperature = .7
    boundary = np.float32(-600. * temperature)
    head = np.array([[0., boundary, np.nextafter(boundary, np.float32(direction))]], dtype=np.float32)
    for gap in (np.nextafter(600., 0.), 600.):
        compare(head, temperature=temperature, top_k=2, max_logit_gap=float(gap))


def test_strided_signed_zero_inputs_remain_byte_exact():
    head = np.array([[-0., 0., -1., -2., -0., 0., -np.inf]], dtype=np.float32)
    for value in (head, head[:, ::-1], head[:, ::2]):
        compare(value, top_k=3)


def test_underflow_warning_not_hidden_by_top_k_selection():
    tiny = np.nextafter(np.float32(0), np.float32(1))
    head = np.array([[0., -tiny]], dtype=np.float32)
    messages = []
    for function in (stable_support_filter, partition_support_filter):
        with warnings.catch_warnings(record=True) as caught, np.errstate(under='warn'):
            warnings.simplefilter('always')
            function(head, temperature=1e308, top_k=1)
        messages.append([(w.category, str(w.message)) for w in caught])
    assert messages[0] and messages[0] == messages[1]
