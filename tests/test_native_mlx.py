"""Optional installed binary and independently identified sampler regressions."""
import copy
import ctypes
import hmac
import json
from pathlib import Path
import random
import shutil
import sys

import numpy as np
import pytest

native_module = pytest.importorskip('keyprint_native')
from keyprint.experimental.native_mlx import NativeCandidate, NativePublicCandidate, VERSION, _NativeProfile
from keyprint.experimental.fast_reporting import build_report
from keyprint._engine.research.keyprint_candidate_v3.adapter import Candidate
from keyprint import Keyprint
from test_fast_mlx import head, assert_parity

sys.path.insert(0, str(Path(__file__).parents[1] / 'tools'))
try:
    from commitment_parity import observe_native_vectors, legacy_receipt_projection
finally:
    sys.path.pop(0)


@pytest.fixture(autouse=True)
def native_vectors():
    with observe_native_vectors():
        yield


def test_packaged_binary_full_digests_and_identity():
    native=native_module.NativePRF()
    key=bytes(range(32));prefix=b'prefix\x00\xff'
    suffixes=[b'',b'a\x00b','你好'.encode(),bytes(4096)]
    expected=b''.join(hmac.digest(key,prefix+layer.to_bytes(4,'big')+s,'sha256') for s in suffixes for layer in range(30))
    assert native.digests(key,prefix,suffixes)==expected
    identity=native.identity
    assert identity['build']['dynamic_dependencies']==['/usr/lib/libSystem.B.dylib']
    assert identity['build']['minimum_macos']=='11.0'
    assert identity['openssl_version'].startswith('OpenSSL 3.6.4')
    identity['build']['files'].clear()
    assert native.identity['build']['files']


@pytest.mark.parametrize('filename',['_native.dylib','_native.c','_select.c','__init__.py','licenses/OpenSSL.txt'])
def test_corruption_rejected_before_library_load(monkeypatch,tmp_path,filename):
    source=Path(native_module.__file__).parent
    shutil.copytree(source,tmp_path/'keyprint_native')
    root=tmp_path/'keyprint_native';target=root/filename
    target.write_bytes(target.read_bytes()+b'\ncorruption')
    monkeypatch.setattr(native_module,'__file__',str(root/'__init__.py'))
    monkeypatch.setattr(ctypes,'CDLL',lambda *a,**k:pytest.fail('corrupt library loaded'))
    with pytest.raises(RuntimeError,match='integrity checks'):native_module.NativePRF()


@pytest.mark.parametrize('condition',['ordinary','marked'])
@pytest.mark.parametrize('size',[10,63,64,100,1001])
def test_supplied_head_sampling_and_context_lifecycle_match(condition,size):
    ref=Candidate(top_k=size);native=NativeCandidate(ref);key=bytes(range(32))
    # Arbitrary vocabulary ranges contain isolated UTF-8 continuation bytes.
    # Use complete ASCII carriers so this test isolates sampler parity.
    tokens=[i for i,piece in enumerate(ref._base._binding.token_bytes)
            if piece and all(32<=byte<127 for byte in piece)][:size]
    assert len(tokens)==size
    a=ref.pipeline(key,condition=condition);b=native.pipeline(key,condition=condition)
    ra,rb=random.Random(42),random.Random(42)
    try:
        for _ in range(8):
            raw=head(tokens)
            assert a.step(raw,ra.getrandbits).token_id==b.step(raw,rb.getrandbits).token_id
        a.finish();b.finish();assert_parity(a,b)
        assert all(not carrier.session.profile._state for carrier in b._raw.carriers)
    finally:a.close();b.close()


def test_native_table_batches_large_explicit_label_sets():
    ref=Candidate()._base._binding.profile
    profile=_NativeProfile(ref,native_module.NativePRF())
    labels=[f'label-{i}'.encode() for i in range(1003)]
    key=bytes(range(32));context=(b'prior',)
    assert profile.table(key,context,labels)=={label:ref.bits(key,context,label) for label in labels}


@pytest.mark.parametrize('failure', ['nan', 'padded_inf', 'wrong_dtype', 'invalid_utf8'])
def test_filter_and_post_commit_failures_match_reference_receipts(failure):
    ref = Candidate()
    native = NativeCandidate(ref)
    raw = head([32])
    if failure == 'nan': raw[0, 32] = np.nan
    elif failure == 'padded_inf': raw[0, -1] = np.inf
    elif failure == 'wrong_dtype': raw = raw.astype(np.float64)
    else:
        token = next(i for i, piece in enumerate(ref._base._binding.token_bytes) if piece == b'\xb5')
        raw = head([token])
    observations = []
    for candidate in (ref, native):
        pipeline = candidate.pipeline(bytes(range(32)), condition='ordinary')
        draws = []
        rng = random.Random(42)
        def bits(count):
            draws.append(count)
            return rng.getrandbits(count)
        try:
            with pytest.raises((ValueError, TypeError, UnicodeDecodeError)) as caught:
                pipeline.step(raw, bits)
            observations.append((type(caught.value), pipeline.committed_token_ids,
                                 legacy_receipt_projection(pipeline.receipt(), getattr(pipeline._raw, '_observed_vector_packets', []))['shared_filter_attempts'] if candidate is native else pipeline._raw.filter_attempts, draws))
            assert pipeline._raw._terminal
        finally:
            pipeline.close()
    assert observations[0] == observations[1]
    assert observations[0][2][-1]['status'] == (
        'failed_after_commit' if failure == 'invalid_utf8' else 'failed_before_commit')


def test_modified_binary_identity_is_not_accepted():
    native=NativeCandidate(Candidate())
    target=native.identity
    target['specification']['execution']['native_prf']['build']['files']['_native.dylib']='0'*64
    from keyprint._engine.research.keyprint_candidate_v3.adapter import digest
    target['runtime_profile_sha256']=digest(target['specification'])
    with pytest.raises(ValueError,match='execution source mismatch'):
        build_report('error',target_identity=target,payload={})


def test_report_is_separately_named_and_no_acceptance_transfers():
    candidate=NativePublicCandidate(Keyprint(key=bytes(range(32)))._candidate)
    report=candidate.score_literal('A short example text for inspection.',bytes(range(32)))
    assert report['schema']=='keyprint.experimental-native-report.v2'
    assert report['target_identity']['version']==VERSION
    assert report['verdict'] is None
    assert report['all_reporting_surfaces_accepted'] is False
