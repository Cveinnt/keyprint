from dataclasses import FrozenInstanceError
from fractions import Fraction as F
from pathlib import Path
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).parents[1]/'tools'))
from paced_source_session import PacedProfile,PacedSourceSession,SamplingFailure,Config,SourceRequest
from keyprint._engine.legacy._impl.research.grouped_canonical_prototype import Profile
from keyprint._engine.research.keyprint_exact_categorical_v2 import integer_distribution

KEY=bytes(range(32))


def profile(pieces=(b'a',b'b',None),**kwargs):
    return PacedProfile(pieces,tokenizer_identity='test-bytes-v1',eos_ids=[len(pieces)-1],**kwargs)


def probs(prepared):
    return {i:F(w,prepared.distribution.total) for i,w in zip(prepared.token_ids,prepared.distribution.weights)}


def ordinary(q):
    d=integer_distribution(tuple(map(float,q)))
    return {i:F(w,d.total) for i,w in enumerate(d.weights) if w}


def choose(session,prepared,token):
    j=prepared.token_ids.index(token)
    point=sum(prepared.distribution.weights[:j])
    draw=session.draw(prepared,lambda count:point)
    assert draw.token_index==token
    return session.commit(prepared,token)


def test_profile_namespace_is_distinct_and_old_profile_rejected():
    p=profile();old=Profile((b'a',b'b',None),tokenizer_identity='test-bytes-v1')
    assert p.digest!=old.digest and p.bits(KEY,(),b'a')!=old.bits(KEY,(),b'a')
    with pytest.raises(TypeError): PacedSourceSession(old,KEY,condition='marked')
    with pytest.raises(ValueError): PacedProfile((b'a',b'eos'),tokenizer_identity='x',eos_ids=[1])


def test_partition_preserves_exact_eos_and_formatting_probabilities():
    s=PacedSourceSession(profile((b'a',b'b',b' \n',None)),KEY,condition='marked')
    q=np.array([.17,.31,.23,.29]);p=s.prepare(q);a,b=ordinary(q),probs(p)
    assert b[2]==a[2] and b[3]==a[3]
    assert sum(b.values())==1 and all(a[i]/2<=b[i]<=2*a[i] for i in a)
    assert s.last_decision['partitioned']
    assert not hasattr(p,'probabilities')  # Cannot masquerade as old float sampling.


def test_actual_integer_normalization_trap_is_avoided():
    # Merely copying raw EOS=.25 while changing other mass would be insufficient.
    s=PacedSourceSession(profile(),KEY,condition='marked')
    q=np.array([.1,.65,.25]);p=s.prepare(q)
    assert probs(p)[2]==ordinary(q)[2]
    assert s.source_receipt()['integer_sampling']


@pytest.mark.parametrize('key',[KEY,*[bytes([i])*32 for i in range(4)]])
def test_proofread_preserves_all_overlapping_whole_piece_continuations(key):
    prefix=b'abcdefghijklmnopqrstuvwx'
    pieces=(prefix,b'A',b' A',b'B',b'C',b'D',None)
    source=(prefix+b'A'+prefix+b'B').decode()
    s=PacedSourceSession(profile(pieces),key,condition='marked',request=SourceRequest('proofread',source))
    p=s.prepare(np.array([1.,0.,0.,0.,0.,0.,0.]));assert s.last_decision['mode']=='startup_ordinary'
    choose(s,p,0)
    q=np.array([0.,.1,.1,.2,.2,.3,.1]);p=s.prepare(q)
    assert set(s.last_decision['protected_token_ids'])=={1,2,3}
    assert s.last_decision['alignment_occurrences']==2
    actual,base=probs(p),ordinary(q)
    assert all(actual[i]==base[i] for i in (1,2,3,6))
    assert s.last_decision['partitioned']


def test_transform_purpose_is_not_silently_routed_to_proofreading():
    s=PacedSourceSession(profile(),KEY,condition='marked',request=SourceRequest('transform','a b'))
    s.prepare(np.array([.4,.4,.2]));assert s.last_decision['mode']=='full_mark'


def test_repeated_context_returns_exact_ordinary_distribution():
    s=PacedSourceSession(profile(config=Config(history=1,layers=30,max_steps=10)),KEY,condition='marked')
    for _ in range(2): choose(s,s.prepare(np.array([1.,0.,0.])),0)
    q=np.array([.4,.4,.2]);p=s.prepare(q)
    assert probs(p)==ordinary(q) and not s.last_decision['partitioned']
    assert not choose(s,p,1).eligible


def test_single_class_and_ordinary_paths_are_exact_identities():
    p=profile((b'a',b' a',None));q=np.array([.2,.6,.2])
    for condition in ('ordinary','marked'):
        s=PacedSourceSession(p,KEY,condition=condition);out=s.prepare(q)
        assert probs(out)==ordinary(q) and not s.last_decision['partitioned']


def test_unicode_byte_fragments_are_not_reencoded():
    p=profile((b'\xc3',b'\xa9',b'\xe4\xb8\xad',None))
    assert p.classes[:3]==(b'\xc3',b'\xa9',b'\xe4\xb8\xad')


def test_eos_commit_closes_session_without_scored_event():
    s=PacedSourceSession(profile(),KEY,condition='marked')
    p=s.prepare(np.array([.2,.3,.5]));assert choose(s,p,2) is None
    with pytest.raises(RuntimeError,match='closed'):s.prepare(np.array([.2,.3,.5]))
    assert not s.source_receipt()['detector_calibrated']


def test_integer_draw_intervals_have_exact_token_multiplicities():
    q=np.array([.5,.25,.25]);counts=[0,0,0]
    for point in range(4):
        s=PacedSourceSession(profile(),KEY,condition='ordinary');p=s.prepare(q)
        draw=s.draw(p,lambda count:point);counts[draw.token_index]+=1
        assert draw.total_weight==4 and draw.transcript[0].bit_count==2
    assert counts==[2,1,1]


def test_rejected_draws_retained_and_accepted_draw_committed_once():
    s=PacedSourceSession(profile(),KEY,condition='ordinary');p=s.prepare(np.array([1/3,1/3,1/3]))
    points=iter([3,1]);d=s.draw(p,lambda count:next(points))
    assert [x.accepted for x in d.transcript]==[False,True]
    with pytest.raises(RuntimeError):s.draw(p,lambda count:0)
    with pytest.raises(ValueError):s.commit(p,0)
    s.commit(p,1)
    with pytest.raises(ValueError):s.commit(p,1)


def test_draw_cap_failure_retains_attempts_and_closes_session():
    s=PacedSourceSession(profile(),KEY,condition='ordinary');p=s.prepare(np.array([1/3]*3))
    with pytest.raises(SamplingFailure) as caught:s.draw(p,lambda count:3,max_draws=2)
    assert len(caught.value.transcript)==2 and caught.value.callback_calls==2
    with pytest.raises(RuntimeError,match='closed'):s.draw(p,lambda count:0)


def test_single_supported_token_uses_no_random_bits():
    s=PacedSourceSession(profile(),KEY,condition='marked');p=s.prepare(np.array([1.,0.,0.]))
    d=s.draw(p,lambda count:pytest.fail('No entropy needed'))
    assert d.token_index==0 and d.transcript==()


def test_immutable_prepared_foreign_step_and_unrecorded_commit_rejected():
    a=PacedSourceSession(profile(),KEY,condition='marked');b=PacedSourceSession(profile(),KEY,condition='marked')
    p=a.prepare(np.array([.3,.4,.3]));other=b.prepare(np.array([.3,.4,.3]))
    with pytest.raises(FrozenInstanceError):p.index=4
    with pytest.raises(ValueError):a.draw(other,lambda count:0)
    with pytest.raises(ValueError):a.commit(p,0)
    with pytest.raises(RuntimeError):a.prepare(np.array([.3,.4,.3]))


def test_owner_and_step_cap_remain_enforced():
    s=PacedSourceSession(profile(config=Config(max_steps=1)),KEY,condition='marked')
    with ThreadPoolExecutor(1) as pool:
        with pytest.raises(RuntimeError):pool.submit(s.prepare,np.array([1.,0.,0.])).result()
    choose(s,s.prepare(np.array([1.,0.,0.])),0)
    with pytest.raises(RuntimeError,match='cap'):s.prepare(np.array([1.,0.,0.]))


def test_unadmitted_numerical_case_fails_closed_without_distribution_fallback():
    s=PacedSourceSession(profile(),KEY,condition='marked')
    with pytest.raises(ValueError):s.prepare(np.array([1.,2.**-950,0.]))
    with pytest.raises(RuntimeError,match='closed'):s.prepare(np.array([.5,.5,0.]))


@pytest.mark.parametrize('slot',range(4))
def test_sparse_full_model_head_keeps_high_ids_and_exact_eos_mass(slot):
    pieces=[None]*248320
    pieces[200003]=b'word';pieces[240117]=b' word';pieces[248319]='中文'.encode()
    p=PacedProfile(pieces,tokenizer_identity='synthetic-wide-not-a-model',eos_ids=[248046])
    s=PacedSourceSession(p,bytes([slot])*32,condition='marked')
    q=np.zeros(248320,dtype=np.float64)
    q[[200003,240117,248319,248046]]=[.17,.31,.23,.29]
    prepared=s.prepare(q);actual=probs(prepared);base=ordinary(q)
    assert set(actual)=={200003,240117,248319,248046}
    assert actual[248046]==base[248046]
    assert all(base[i]/2<=actual[i]<=2*base[i] for i in actual)
    choose(s,prepared,248319)
    assert s._context==('中文'.encode(),)


def test_proofread_startup_expires_without_inventing_source_alignment():
    piece=b'x'*96
    p=profile((piece,b'a',b'b',None))
    s=PacedSourceSession(p,KEY,condition='marked',request=SourceRequest('proofread','unrelated source'))
    first=s.prepare(np.array([1.,0.,0.,0.]));choose(s,first,0)
    s.prepare(np.array([0.,.4,.4,.2]))
    assert s.last_decision['mode']=='full_mark'
    assert s.last_decision['protected_token_ids']==()
    assert s.source_receipt()['startup_exhausted']


@pytest.mark.parametrize('value',[True,-1,1.5])
def test_invalid_random_callback_closes_without_redraw(value):
    s=PacedSourceSession(profile(),KEY,condition='ordinary');p=s.prepare(np.array([1/3]*3))
    with pytest.raises(SamplingFailure) as caught:s.draw(p,lambda count:value)
    assert caught.value.callback_calls==1 and caught.value.transcript==()
    with pytest.raises(RuntimeError,match='closed'):s.draw(p,lambda count:0)


def test_eos_policy_cannot_mutate_under_an_existing_profile_digest():
    p=profile()
    with pytest.raises(FrozenInstanceError):p.eos_ids=frozenset([0])
    assert p.eos_ids==frozenset([2])
