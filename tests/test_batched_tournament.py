import math

import numpy as np
import pytest
from keyprint._engine.legacy._impl.research.byte_trie_numeric import update, MIN_POSITIVE

from keyprint.experimental import batched_tournament as candidate


@pytest.mark.parametrize("size", [1,10,63,64,100,1000])
@pytest.mark.parametrize("pattern", ["zero", "one", "alternating", "random"])
def test_all_layers_match_including_subnormal_floor_and_zero_support(size,pattern):
    rng = np.random.default_rng(size)
    q = rng.random(size)
    if size > 1: q[0] = MIN_POSITIVE
    if size > 2: q[1] = 0
    q /= math.fsum(q.tolist())
    bits = np.zeros((30,size), dtype=np.int8)
    if pattern == "one": bits[:] = 1
    elif pattern == "alternating": bits[:,1::2] = 1
    elif pattern == "random": bits = rng.integers(0,2,size=(30,size),dtype=np.int8)
    value, counters, expected = tuple(q), {}, []
    for layer in bits.tolist():
        value = update(value, layer, counters)
        expected.append((np.asarray(value).tobytes(),counters.copy()))
    actual = []
    before = q.copy()
    with np.errstate(all="raise"):
        candidate.update_layers(q,bits,{},observe=lambda p,c: actual.append((p.tobytes(),c.copy())))
    assert actual == expected
    assert q.tobytes() == before.tobytes()


@pytest.mark.parametrize("kind", ["empty", "negative", "nan", "mass", "bits", "shape", "dtype"])
def test_invalid_inputs_are_rejected(kind):
    q = np.array([.5,.5]); bits = np.zeros((30,2),dtype=np.int8)
    if kind == "empty": q=np.array([]); bits=np.zeros((30,0),dtype=np.int8)
    elif kind == "negative": q[0]=-1
    elif kind == "nan": q[0]=float('nan')
    elif kind == "mass": q[0]=.4
    elif kind == "bits": bits[10,0]=2
    elif kind == "shape": bits=bits[:,:1]
    else: bits=bits.astype(np.float64)
    with pytest.raises(ValueError): candidate.update_layers(q,bits)


@pytest.mark.parametrize("size", [10,63,64,100])
def test_transform_dispatch_preserves_protection_and_exact_mass(size):
    from keyprint._engine.legacy._impl.research.grouped_canonical_prototype import Config, Profile
    from keyprint._engine.legacy._impl.research.token_source_sparse_execution import transform
    profile = Profile([f"label-{i}".encode() for i in range(size+2)],tokenizer_identity="batch-test",config=Config(layers=30))
    q = np.array([.9/size]*size+[.1,0.],dtype=np.float64)
    before = q.copy()
    a, b = {"branch_roundups":0,"partition_roundups":0}, {"branch_roundups":0,"partition_roundups":0}
    expected = transform(q,profile,bytes(range(32)),(b"prior",),[size],a)
    with np.errstate(all="raise"):
        actual = candidate.transform(q,profile,bytes(range(32)),(b"prior",),[size],b)
    assert expected.tobytes()==actual.tobytes()
    assert a==b and q.tobytes()==before.tobytes()
    assert actual[size]==q[size] and actual[-1]==0.
