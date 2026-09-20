import importlib.util
from pathlib import Path

import numpy as np
import pytest
from keyprint._engine.legacy._impl.research.byte_trie_numeric import update, MIN_POSITIVE

spec = importlib.util.spec_from_file_location("tournament_candidate", Path(__file__).parents[1] / "tools/benchmark_tournament.py")
candidate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(candidate)


@pytest.mark.parametrize("values", [[1.], [.5,.5], [MIN_POSITIVE, 1.], [0.,.1,.2,.7], [1e-300,.5,.5]])
@pytest.mark.parametrize("pattern", ["zero", "one", "alternating"])
def test_layerwise_binary64_and_floor_counters_match(values, pattern):
    a, b = tuple(values), np.array(values, dtype=np.float64)
    bits = [0 if pattern == "zero" else 1 if pattern == "one" else i%2 for i in range(len(a))]
    left, right = {}, {}
    with np.errstate(all="raise"):
        for _ in range(30):
            a = update(a, bits, left)
            b = candidate.update_array(b, np.array(bits,dtype=np.int8), right)
            assert np.asarray(a).tobytes() == b.tobytes()
            assert left == right


@pytest.mark.parametrize("values,bits", [([.1,.2],[0,1]), ([float('nan')],[0]), ([-1.,2.],[0,1]), ([1.],[2]), ([0.],[0]), ([],[])])
def test_invalid_inputs_remain_errors(values, bits):
    with pytest.raises(ValueError): update(tuple(values), bits)
    with pytest.raises(ValueError): candidate.update_array(np.array(values,dtype=np.float64), np.array(bits,dtype=np.int8))
